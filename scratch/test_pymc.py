import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import pymc as pm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.stage1_loader import run_stage1
from src.stage2_cleaner import run_stage2
from src.stage3_engineering import run_stage3
from src.stage5_esda import compare_weights

def test_mcmc():
    print("Loading data...")
    df_raw = run_stage1()
    df_clean = run_stage2(df_raw)
    df_eng, gdf_total, gdf_anual, gdf_mensual = run_stage3(df_clean)
    w_selected, w_name, df_comp = compare_weights(gdf_total)
    
    # Run a super fast version of Stage 8 (MCMC)
    from src.stage8_bayesian import build_neighbor_graph
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
    
    homicidios_obs = df_train['homicidios'].values
    sub_idx = df_train['sub_idx'].values
    time_idx = df_train['time_idx'].values
    log_area = df_train['log_area'].values
    
    node_i, node_j = build_neighbor_graph(w_selected)
    
    print("Compiling model...")
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=3)
        sigma_phi = pm.HalfNormal('sigma_phi', sigma=1.0)
        sigma_t = pm.HalfNormal('sigma_t', sigma=1.0)
        
        phi_raw = pm.Normal('phi_raw', mu=0, sigma=1.0, shape=n_zonas)
        phi_centered = phi_raw - pm.math.mean(phi_raw)
        phi = pm.Deterministic('phi', phi_centered * sigma_phi)
        
        pm.Potential(
            'spatial_penalty', 
            -0.5 * pm.math.sum((phi[node_i] - phi[node_j])**2) - 0.01 * pm.math.sum(phi**2)
        )
        
        gamma_raw = pm.Normal('gamma_raw', mu=0, sigma=1.0, shape=n_periodos)
        gamma = pm.Deterministic('gamma', pm.math.cumsum(gamma_raw) * sigma_t)
        
        log_mu = alpha + phi[sub_idx] + gamma[time_idx] + log_area
        mu = pm.math.exp(log_mu)
        alpha_disp = pm.Exponential('alpha_disp', lam=1.0)
        
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
        
        print("Starting fast sampling (10 draws, 10 tune)...")
        trace = pm.sample(draws=10, tune=10, chains=1, cores=1, target_accept=0.8, random_seed=42)
        print("Sampling completed successfully!")

if __name__ == "__main__":
    test_mcmc()
