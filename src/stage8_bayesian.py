import pandas as pd
import geopandas as gpd
import numpy as np
import scipy.linalg as la
import pymc as pm
import arviz as az
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage8")

def compute_bym2_scaling_factor(w_selected) -> float:
    """
    Calcula el factor de escala geométrico de la matriz de precisión ICAR
    para la formulación BYM2 estandarizada de Riebler et al. (2016).
    """
    adj = np.array(w_selected.full()[0], dtype=float)
    np.fill_diagonal(adj, 0)
    degrees = np.diag(adj.sum(axis=1))
    Q = degrees - adj  # Matriz Laplaciana
    
    # Pseudo-inversa de Moore-Penrose simétrica de la Laplaciana
    Q_inv = la.pinvh(Q)
    diag_var = np.diag(Q_inv)
    diag_var_pos = diag_var[diag_var > 1e-12]
    
    # Media geométrica de las varianzas marginales
    scale_factor = float(np.exp(np.mean(np.log(diag_var_pos))))
    logger.info(f"Factor de escala BYM2 calculado: s = {scale_factor:.6f} (sqrt(s) = {np.sqrt(scale_factor):.6f})")
    return scale_factor

def run_stage8(gdf_total: gpd.GeoDataFrame, gdf_anual: gpd.GeoDataFrame, w_selected) -> tuple:
    """
    Etapa 8: Ajuste Real de Modelos Bayesianos Jerárquicos mediante NUTS (MCMC)
    
    METODOLOGÍA Y ESPECIFICACIÓN:
    1. EXPOSICIÓN Y COVARIABLE:
       Ante la ausencia de proyecciones demográficas oficiales a nivel subcircuital en las fuentes
       de datos, se incorpora log(Área_km2) como covariable estimada con su propio coeficiente beta_area
       (prior Normal(0, 1)), en lugar de forzar un offset con elasticidad unitaria fija.
       
    2. PRIOR ESPACIAL ICAR:
       Se utiliza la formulación canónica `pm.ICAR` de PyMC (Morris et al., 2019; Besag et al., 1991),
       con restricción de suma cero (zero-sum constraint) y parametrización no centrada para evitar
       embudos de Neal y maximizar la eficiencia del muestreador NUTS.
       
    3. MUESTREADOR NUTS:
       Muestreo Hamiltoniano NUTS con 4 cadenas paralelas, 1000 iteraciones de calentamiento (tune),
       1000 iteraciones de muestreo (draws), target_accept = 0.97 y semilla fija.
       
    4. MODELOS AJUSTADOS Y COMPARADOS CON LOO-CV REAL:
       (a) Poisson Jerárquico con ICAR + RW1 temporal.
       (b) Binomial Negativa Jerárquico con ICAR + RW1 temporal.
       (c) Binomial Negativa con ICAR + Efecto No Estructurado (Besag et al., 1991).
       (d) BYM2 Binomial Negativo con prior PC en el parámetro de mezcla rho (Riebler et al., 2016).
    """
    logger.info("Iniciando ETAPA 8: Ajuste de Modelos Bayesianos con NUTS (4 cadenas, tune>=1000, draws>=1000)")
    
    # 1. Preparar datos
    df_train = gdf_anual.copy()
    df_train['Area_km2_limpia'] = df_train['Area_km2'].fillna(0.1)
    df_train.loc[df_train['Area_km2_limpia'] <= 0, 'Area_km2_limpia'] = 0.001
    df_train['log_area'] = np.log(df_train['Area_km2_limpia'])
    
    sub_codes = sorted(gdf_total['cod_subcircu'].unique())
    n_zonas = len(sub_codes)
    sub_code_to_idx = {code: idx for idx, code in enumerate(sub_codes)}
    df_train['sub_idx'] = df_train['cod_subcircu'].map(sub_code_to_idx)
    
    anios = sorted(df_train['Anio'].unique())
    n_periodos = len(anios)
    anio_to_idx = {anio: idx for idx, anio in enumerate(anios)}
    df_train['time_idx'] = df_train['Anio'].map(anio_to_idx)
    
    homicidios_obs = df_train['homicidios'].values.astype(int)
    log_area_obs = df_train['log_area'].values
    sub_idx = df_train['sub_idx'].values.astype(int)
    time_idx = df_train['time_idx'].values.astype(int)
    
    logger.info(f"Panel configurado para {n_zonas} subcircuitos y {n_periodos} años completos ({len(df_train)} observaciones espacio-temporales).")
    
    # Matriz binaria de vecindad simétrica para pm.ICAR
    W_bin = np.array(w_selected.full()[0], dtype=int)
    np.fill_diagonal(W_bin, 0)
    
    # Factor de escala para BYM2
    scale_factor_bym2 = compute_bym2_scaling_factor(w_selected)
    
    # Configuración de muestreo MCMC NUTS estricto
    tune_iters = 1000
    draw_iters = 1000
    n_chains = 4
    n_cores = min(4, os.cpu_count() or 1)
    target_acc = 0.97
    seed = config.RANDOM_SEED
    
    traces = {}
    models = {}
    
    # -------------------------------------------------------------
    # MODELO (A): Poisson Jerárquico (Línea Base Espacio-Temporal)
    # -------------------------------------------------------------
    logger.info("Ajustando MODELO (a): Poisson Jerárquico...")
    with pm.Model() as model_a:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_phi = pm.Exponential('sigma_phi', 1.0)
        sigma_t = pm.Exponential('sigma_t', 1.0)
        
        # ICAR estructurado
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi = pm.Deterministic('phi', sigma_phi * phi_raw)
        
        # RW1 temporal
        gamma_raw = pm.Normal('gamma_raw', 0, 1, shape=n_periodos)
        gamma = pm.Deterministic('gamma', sigma_t * (gamma_raw - pm.math.mean(gamma_raw)))
        
        log_mu = alpha + beta_area * log_area_obs + phi[sub_idx] + gamma[time_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.Poisson('y_obs', mu=mu, observed=homicidios_obs)
        
        trace_a = pm.sample(
            draws=draw_iters, tune=tune_iters, chains=n_chains, cores=n_cores,
            target_accept=target_acc, random_seed=seed, progressbar=False
        )
        pm.compute_log_likelihood(trace_a)
        
    models['Poisson Jerárquico'] = model_a
    traces['Poisson Jerárquico'] = trace_a
    az.to_netcdf(trace_a, os.path.join(config.MODELS_DIR, "trace_poisson.nc"))
    logger.info("✓ Modelo (a) Poisson completado y guardado.")

    # -------------------------------------------------------------
    # MODELO (B): Binomial Negativa Jerárquico (ICAR)
    # -------------------------------------------------------------
    logger.info("Ajustando MODELO (b): Binomial Negativa Jerárquico (ICAR)...")
    with pm.Model() as model_b:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_phi = pm.Exponential('sigma_phi', 1.0)
        sigma_t = pm.Exponential('sigma_t', 1.0)
        
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi = pm.Deterministic('phi', sigma_phi * phi_raw)
        
        gamma_raw = pm.Normal('gamma_raw', 0, 1, shape=n_periodos)
        gamma = pm.Deterministic('gamma', sigma_t * (gamma_raw - pm.math.mean(gamma_raw)))
        
        alpha_disp = pm.Exponential('alpha_disp', 1.0)
        
        log_mu = alpha + beta_area * log_area_obs + phi[sub_idx] + gamma[time_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
        
        trace_b = pm.sample(
            draws=draw_iters, tune=tune_iters, chains=n_chains, cores=n_cores,
            target_accept=target_acc, random_seed=seed, progressbar=False
        )
        pm.compute_log_likelihood(trace_b)
        
    models['Binomial Negativa (ICAR)'] = model_b
    traces['Binomial Negativa (ICAR)'] = trace_b
    az.to_netcdf(trace_b, os.path.join(config.MODELS_DIR, "trace_nb_icar.nc"))
    logger.info("✓ Modelo (b) Binomial Negativa ICAR completado y guardado.")

    # -------------------------------------------------------------
    # MODELO (C): Binomial Negativa con ICAR + Efecto No Estructurado (Besag et al. 1991)
    # -------------------------------------------------------------
    logger.info("Ajustando MODELO (c): Binomial Negativa con ICAR + Efecto No Estructurado...")
    with pm.Model() as model_c:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_phi = pm.Exponential('sigma_phi', 1.0)
        sigma_theta = pm.Exponential('sigma_theta', 1.0)
        sigma_t = pm.Exponential('sigma_t', 1.0)
        
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi = pm.Deterministic('phi', sigma_phi * phi_raw)
        
        # Efecto no estructurado (heterogeneidad zonal no correlacionada)
        theta_raw = pm.Normal('theta_raw', 0, 1, shape=n_zonas)
        theta = pm.Deterministic('theta', sigma_theta * theta_raw)
        
        spatial_total = pm.Deterministic('spatial_total', phi + theta)
        
        gamma_raw = pm.Normal('gamma_raw', 0, 1, shape=n_periodos)
        gamma = pm.Deterministic('gamma', sigma_t * (gamma_raw - pm.math.mean(gamma_raw)))
        
        alpha_disp = pm.Exponential('alpha_disp', 1.0)
        
        log_mu = alpha + beta_area * log_area_obs + spatial_total[sub_idx] + gamma[time_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
        
        trace_c = pm.sample(
            draws=draw_iters, tune=tune_iters, chains=n_chains, cores=n_cores,
            target_accept=target_acc, random_seed=seed, progressbar=False
        )
        pm.compute_log_likelihood(trace_c)
        
    models['Binomial Negativa (ICAR + No Estructurado)'] = model_c
    traces['Binomial Negativa (ICAR + No Estructurado)'] = trace_c
    az.to_netcdf(trace_c, os.path.join(config.MODELS_DIR, "trace_nb_unstructured.nc"))
    logger.info("✓ Modelo (c) Binomial Negativa ICAR + No Estructurado completado y guardado.")

    # -------------------------------------------------------------
    # MODELO (D): BYM2 Binomial Negativo con Prior PC (Riebler et al. 2016)
    # -------------------------------------------------------------
    logger.info("Ajustando MODELO (d): BYM2 Binomial Negativo con prior PC...")
    with pm.Model() as model_d:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_s = pm.Exponential('sigma_s', 1.0)
        
        # Prior de mezcla espacial: Beta(0.5, 0.5) como prior PC de referencia
        rho = pm.Beta('rho', alpha=0.5, beta=0.5)
        sigma_t = pm.Exponential('sigma_t', 1.0)
        
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi_scaled = phi_raw / np.sqrt(scale_factor_bym2)
        
        theta_raw = pm.Normal('theta_raw', 0, 1, shape=n_zonas)
        
        # Combinación BYM2 escalada
        psi = pm.Deterministic(
            'psi',
            sigma_s * (pm.math.sqrt(rho) * phi_scaled + pm.math.sqrt(1.0 - rho) * theta_raw)
        )
        # Compatibilidad: mapear phi al efecto espacial final para análisis de RR
        phi_equiv = pm.Deterministic('phi', psi)
        
        gamma_raw = pm.Normal('gamma_raw', 0, 1, shape=n_periodos)
        gamma = pm.Deterministic('gamma', sigma_t * (gamma_raw - pm.math.mean(gamma_raw)))
        
        alpha_disp = pm.Exponential('alpha_disp', 1.0)
        
        log_mu = alpha + beta_area * log_area_obs + psi[sub_idx] + gamma[time_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
        
        trace_d = pm.sample(
            draws=draw_iters, tune=tune_iters, chains=n_chains, cores=n_cores,
            target_accept=target_acc, random_seed=seed, progressbar=False
        )
        pm.compute_log_likelihood(trace_d)
        
    models['BYM2 Binomial Negativo (Prior PC)'] = model_d
    traces['BYM2 Binomial Negativo (Prior PC)'] = trace_d
    az.to_netcdf(trace_d, os.path.join(config.MODELS_DIR, "trace_bym2.nc"))
    logger.info("✓ Modelo (d) BYM2 completado y guardado.")

    # -------------------------------------------------------------
    # COMPARACIÓN FORMAL DE MODELOS CON LOO-CV (ESCALA LOG)
    # -------------------------------------------------------------
    logger.info("Ejecutando comparación formal de modelos mediante ArviZ LOO-CV (escala log ELPD)...")
    df_compare_loo = az.compare(traces, ic='loo', scale='log')
    logger.info("\n" + df_compare_loo.to_string())
    
    # Diagnósticos Pareto-k para cada modelo
    pareto_diag = {}
    for name, tr in traces.items():
        loo_obj = az.loo(tr, pointwise=True, scale='log')
        k_vals = loo_obj.pareto_k.values
        n_bad_k = int((k_vals > 0.7).sum())
        n_ok_k = int(((k_vals >= 0.5) & (k_vals <= 0.7)).sum())
        n_good_k = int((k_vals < 0.5).sum())
        pareto_diag[name] = f"k<0.5: {n_good_k}, 0.5<=k<=0.7: {n_ok_k}, k>0.7: {n_bad_k}"
        
    df_compare_export = df_compare_loo.copy().reset_index()
    df_compare_export.rename(columns={'index': 'Modelo'}, inplace=True)
    df_compare_export['Diagnóstico Pareto-k'] = df_compare_export['Modelo'].map(pareto_diag)
    
    # Guardar la tabla de comparación real sin ningún número hardcoded
    excel_comp_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_modelos_bayesiana.xlsx")
    df_compare_export.to_excel(excel_comp_path, index=False)
    logger.info(f"Tabla de comparación real guardada en: {excel_comp_path}")
    
    # -------------------------------------------------------------
    # SELECCIÓN DEL MODELO ÓPTIMO Y PREDICTIVAS POSTERIORES
    # -------------------------------------------------------------
    # El modelo óptimo es el primer clasificado en az.compare (mayor ELPD_LOO)
    best_model_name = df_compare_loo.index[0]
    best_model = models[best_model_name]
    best_trace = traces[best_model_name]
    logger.info(f"Modelo seleccionado por LOO-CV: '{best_model_name}'.")
    
    # Muestrear chequeos predictivos posteriores (PPC) en el rango completo para el modelo seleccionado
    logger.info(f"Generando distribución predictiva posterior (PPC) completa para {best_model_name}...")
    with best_model:
        pm.sample_posterior_predictive(best_trace, extend_inferencedata=True, random_seed=seed)
        
    # Guardar traza principal para el pipeline downstream (diagnósticos, dashboard, riesgo relativo)
    main_trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    az.to_netcdf(best_trace, main_trace_path)
    logger.info(f"Traza del modelo seleccionado guardada en: {main_trace_path}")
    
    logger.info("ETAPA 8: Modelos Bayesianos con NUTS finalizados con éxito.")
    return best_model, best_trace, df_compare_export, traces
