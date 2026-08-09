import pandas as pd
import geopandas as gpd
import numpy as np
import pymc as pm
import arviz as az
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage8")

def build_neighbor_graph(w_selected) -> tuple:
    """Extrae las listas de nodos conectados de la matriz de pesos espaciales"""
    node_i = []
    node_j = []
    
    # Estandarizar índices numéricos secuenciales
    for node, neighbors in w_selected.neighbors.items():
        for neighbor in neighbors:
            if node < neighbor:  # Evitar duplicar aristas (grafo no dirigido)
                node_i.append(node)
                node_j.append(neighbor)
                
    return np.array(node_i, dtype=int), np.array(node_j, dtype=int)

def run_stage8(gdf_total: gpd.GeoDataFrame, gdf_anual: gpd.GeoDataFrame, w_selected) -> tuple:
    """
    Etapa 8: Ajuste del Modelo Bayesiano (Ajuste Real mediante PyMC y ADVI)
    - Calcula el offset de casos esperados E_i basado en riesgo uniforme proporcional al área del subcircuito.
    - Compila un modelo jerárquico binomial negativo con prior de vecindad Queen espacial (ICAR) y Random Walk 1 temporal (RW1).
    - Ajusta el modelo probabilístico real en los datos usando Inferencia Variacional ADVI (10,000 iteraciones).
    - Extrae la traza de ArviZ compatible con PPC y diagnósticos, calculando el log-likelihood.
    """
    logger.info("Iniciando ETAPA 8: Ajuste del Modelo Bayesiano Real (PyMC ADVI)")
    
    # 1. Preparar datos
    df_train = gdf_anual.copy()
    df_train['Area_km2_limpia'] = df_train['Area_km2'].fillna(0.1)
    df_train.loc[df_train['Area_km2_limpia'] <= 0, 'Area_km2_limpia'] = 0.001
    
    # 2. Calcular Casos Esperados E_i bajo riesgo uniforme por km2 en Guayaquil
    total_homicidios = df_train['homicidios'].sum()
    total_area = df_train['Area_km2_limpia'].sum()
    mean_rate = total_homicidios / total_area if total_area > 0 else 1.0
    
    df_train['expected_cases'] = df_train['Area_km2_limpia'] * mean_rate
    log_E = np.log(df_train['expected_cases'].values)
    
    sub_codes = sorted(gdf_total['cod_subcircu'].unique())
    n_zonas = len(sub_codes)
    sub_code_to_idx = {code: idx for idx, code in enumerate(sub_codes)}
    df_train['sub_idx'] = df_train['cod_subcircu'].map(sub_code_to_idx)
    
    anios = sorted(df_train['Anio'].unique())
    n_periodos = len(anios)
    anio_to_idx = {anio: idx for idx, anio in enumerate(anios)}
    df_train['time_idx'] = df_train['Anio'].map(anio_to_idx)
    
    homicidios_obs = df_train['homicidios'].values
    sub_idx = df_train['sub_idx'].values
    time_idx = df_train['time_idx'].values
    
    logger.info(f"Modelo configurado para {n_zonas} subcircuitos y {n_periodos} años.")
    logger.info(f"Total de observaciones espacio-temporales: {len(df_train)}")
    
    # Construir grafo de vecinos Queen
    node_i, node_j = build_neighbor_graph(w_selected)
    
    # 3. Compilar el modelo en PyMC
    logger.info("Compilando el modelo probabilístico en PyMC...")
    with pm.Model() as model:
        # Intercepto global (tasa de homicidios basal)
        alpha = pm.Normal('alpha', mu=0, sigma=3)
        
        # Hiper-priors para desvíos estándar de los efectos estructurales
        sigma_phi = pm.HalfNormal('sigma_phi', sigma=1.0)
        sigma_t = pm.HalfNormal('sigma_t', sigma=1.0)
        
        # Efecto espacial aleatorio estructurado con prior ICAR (no centrado)
        phi_raw = pm.Normal('phi_raw', mu=0, sigma=1.0, shape=n_zonas)
        phi_centered = phi_raw - pm.math.mean(phi_raw)
        phi = pm.Deterministic('phi', phi_centered * sigma_phi)
        
        # Penalización espacial ICAR (pairwise neighbors squared difference)
        pm.Potential(
            'spatial_penalty', 
            -0.5 * pm.math.sum((phi_centered[node_i] - phi_centered[node_j])**2) - 0.01 * pm.math.sum(phi_centered**2)
        )
        
        # Tendencia temporal aleatoria con caminata aleatoria de orden 1 (RW1)
        gamma_raw = pm.Normal('gamma_raw', mu=0, sigma=1.0, shape=n_periodos)
        gamma = pm.Deterministic('gamma', pm.math.cumsum(gamma_raw) * sigma_t)
        
        # log(mu_it) = alpha + phi_i + gamma_t + log(E_i)
        log_mu = alpha + phi[sub_idx] + gamma[time_idx] + log_E
        mu = pm.math.exp(log_mu)
        
        # Parámetro de dispersión de la verosimilitud Binomial Negativa
        alpha_disp = pm.Exponential('alpha_disp', lam=1.0)
        
        # Verosimilitud (Likelihood) observada
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
        
        # 4. Ajuste real mediante ADVI (Inferencia Variacional)
        logger.info("Ajustando el modelo mediante ADVI (10,000 iteraciones)...")
        approx = pm.fit(method='advi', n=10000, random_seed=42)
        
        # Extraer muestras de la aproximación de la posterior
        logger.info("Muestreando de la aproximación de la posterior (600 draws)...")
        trace = approx.sample(draws=600)
        
        # 5. Calcular log-verosimilitud para WAIC y LOO-CV
        logger.info("Calculando log-verosimilitud por muestra...")
        pm.compute_log_likelihood(trace)
        
        # 5.5 Muestrear de la distribución predictiva posterior (PPC)
        logger.info("Muestreando de la distribución predictiva posterior (PPC)...")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)
        
    # 6. Guardar la traza en outputs/models/pymc_trace.nc
    trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    az.to_netcdf(trace, trace_path)
    
    logger.info(f"Traza de inferencia real y PPC guardada con éxito en: {trace_path}")
    logger.info("ETAPA 8: Modelo Bayesiano finalizado con éxito.")
    
    return model, trace
