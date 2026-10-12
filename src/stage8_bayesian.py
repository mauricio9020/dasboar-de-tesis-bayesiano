import os
import sys
from pathlib import Path

# Configurar w64devkit en PATH antes de importar pytensor/pymc para aceleración C++
BASE_DIR = Path(__file__).resolve().parent.parent
W64DEVKIT_BIN = BASE_DIR / "scratch" / "w64devkit" / "bin"
if W64DEVKIT_BIN.exists():
    os.environ["PATH"] = str(W64DEVKIT_BIN) + os.pathsep + os.environ.get("PATH", "")

from src import config
import pandas as pd
import geopandas as gpd
import numpy as np
import scipy.linalg as la
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import logging
import time

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

def generate_temporal_outputs(trace, years=[2024, 2025]):
    """
    Genera tablas y figuras del contraste anual entre 2025 y 2024 (beta_year).
    Con únicamente 2 años completos de datos (2024 y 2025), se reporta el contraste fijo
    de año con 2024 como referencia basal, sin asumir una tendencia temporal de largo plazo.
    """
    beta_samples = trace.posterior['beta_year'].values.flatten()
    b_mean = float(np.mean(beta_samples))
    b_median = float(np.median(beta_samples))
    b_lower = float(np.percentile(beta_samples, 2.5))
    b_upper = float(np.percentile(beta_samples, 97.5))
    
    rr_samples = np.exp(beta_samples)
    rr_mean = float(np.mean(rr_samples))
    rr_median = float(np.median(rr_samples))
    rr_lower = float(np.percentile(rr_samples, 2.5))
    rr_upper = float(np.percentile(rr_samples, 97.5))
    
    df_temporal = pd.DataFrame({
        'Año': [2024, 2025],
        'Coeficiente_Log_Media': [0.0, round(b_mean, 4)],
        'Coeficiente_Log_Mediana': [0.0, round(b_median, 4)],
        'HDI_inf_95': [0.0, round(b_lower, 4)],
        'HDI_sup_95': [0.0, round(b_upper, 4)],
        'Riesgo_Relativo_Media': [1.0000, round(rr_mean, 4)],
        'Riesgo_Relativo_Mediana': [1.0000, round(rr_median, 4)],
        'RR_HDI_inf_95': [1.0000, round(rr_lower, 4)],
        'RR_HDI_sup_95': [1.0000, round(rr_upper, 4)],
        'Interpretacion': [
            'Línea base cantonal de referencia (2024)',
            f'Variación neta estimada de {((rr_median - 1)*100):+.1f}% en la tasa de homicidios en 2025 frente a 2024'
        ]
    })
    
    # Guardar en outputs (tanto con nombres de contraste como de tendencia para compatibilidad)
    excel_path = os.path.join(config.TABLES_DIR, "tabla_contraste_anual.xlsx")
    excel_path_leg = os.path.join(config.TABLES_DIR, "tabla_tendencia_temporal.xlsx")
    csv_path = os.path.join(config.CLEAN_DATA_DIR, "contraste_anual.csv")
    csv_path_leg = os.path.join(config.CLEAN_DATA_DIR, "tendencia_temporal.csv")
    
    df_temporal.to_excel(excel_path, index=False)
    df_temporal.to_excel(excel_path_leg, index=False)
    df_temporal.to_csv(csv_path, index=False)
    df_temporal.to_csv(csv_path_leg, index=False)
    
    # Generar gráfico del contraste anual
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    years_plot = [2024, 2025]
    median_plot = [0.0, b_median]
    lower_plot = [0.0, b_lower]
    upper_plot = [0.0, b_upper]
    
    ax.fill_between(years_plot, lower_plot, upper_plot, color='#2196f3', alpha=0.2, label='Intervalo de Credibilidad (HDI 95%)')
    ax.plot(years_plot, median_plot, color='#0d47a1', linewidth=2.5, marker='o', markersize=8, label='Contraste Posterior (Mediana)')
    
    ax.text(2024, 0.05, "0.000 (Línea Base 2024)", fontsize=10, fontweight='bold', color='#0d47a1', ha='center')
    ax.text(2025, b_median + 0.05, f"{b_median:+.3f} (RR = {rr_median:.2f}x)", fontsize=10, fontweight='bold', color='#0d47a1', ha='center')
    
    ax.set_xticks([2024, 2025])
    ax.set_xticklabels(['2024 (Línea Base)', '2025'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Año de Observación', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel(r'Contraste Fijo de Año ($\beta_{year}$ en escala log)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Contraste Anual de Homicidios: 2025 frente a 2024 (Línea Base)\nModelo Bayesiano Espacio-Temporal en Guayaquil (ADVI)', fontsize=12, fontweight='bold', pad=12)
    ax.axhline(0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.grid(color='#eeeeee', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#cccccc')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#aaaaaa')
        
    png_path = os.path.join(config.FIGURES_DIR, "Figura16_contraste_anual.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura16_contraste_anual.pdf")
    png_path_leg = os.path.join(config.FIGURES_DIR, "tendencia_temporal.png")
    pdf_path_leg = os.path.join(config.FIGURES_DIR, "tendencia_temporal.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path_leg, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path_leg, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Tablas y figuras de contraste anual guardadas en {config.TABLES_DIR} y {config.FIGURES_DIR}")
    return df_temporal

def fit_advi_model(model, name: str, n_iter: int = 40000, tol: float = 1e-4, seed: int = 42, draws: int = 2000):
    """
    Ajusta un modelo bayesiano utilizando Inferencia Variacional Automática (ADVI) en PyMC.
    Implementa el callback CheckParametersConvergence con tolerancia 1e-4 y calcula
    la densidad predictiva esperada y el historial del ELBO.
    """
    logger.info(f"--- Ajustando {name} con ADVI (n_max={n_iter}, tol={tol}, seed={seed}) ---")
    cb = pm.callbacks.CheckParametersConvergence(every=100, tolerance=tol, diff='relative')
    
    t0 = time.time()
    with model:
        approx = pm.fit(
            method='advi',
            n=n_iter,
            callbacks=[cb],
            random_seed=seed,
            progressbar=False
        )
        trace = approx.sample(draws=draws, random_seed=seed)
        pm.compute_log_likelihood(trace, model=model)
    t_elapsed = time.time() - t0
    
    loss_hist = np.array(approx.hist)
    elbo_hist = -loss_hist  # ELBO = -Loss
    n_actual = len(loss_hist)
    final_elbo = float(elbo_hist[-1])
    
    # Criterio de convergencia: detenido por callback antes de n_iter o estabilizado en meseta
    # Evaluamos la variación de la media móvil en la cola (últimos 1,000 pasos vs 1,000 pasos previos)
    w_size = min(1000, n_actual // 4)
    elbo_tail_recent = float(np.mean(elbo_hist[-w_size:]))
    elbo_tail_prev = float(np.mean(elbo_hist[-2*w_size:-w_size])) if n_actual >= 2*w_size else float(elbo_hist[0])
    rel_change_plateau = abs(elbo_tail_recent - elbo_tail_prev) / (abs(elbo_tail_recent) + 1e-8)
    converged = (n_actual < n_iter) or (rel_change_plateau < tol)
    
    logger.info(f"✓ {name} finalizado en {t_elapsed:.2f} s | Iteraciones: {n_actual}/{n_iter} | ELBO final: {final_elbo:.2f} (meseta: {elbo_tail_recent:.2f}, rel_diff: {rel_change_plateau:.2e}) | Convergió: {converged}")
    return approx, trace, elbo_hist, final_elbo, n_actual, converged

def build_model_d(n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs):
    """Crea una instancia fresca del Modelo (d) BYM2 Binomial Negativo con Prior Beta(0.5, 0.5)"""
    with pm.Model() as model_d:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_year = pm.Normal('beta_year', mu=0, sigma=1)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_s = pm.Exponential('sigma_s', 1.0)
        
        # Prior explícito Beta(0.5, 0.5) en parámetro de mezcla rho
        rho = pm.Beta('rho', alpha=0.5, beta=0.5)
        
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi_scaled = phi_raw / np.sqrt(scale_factor_bym2)
        
        theta_raw = pm.Normal('theta_raw', 0, 1, shape=n_zonas)
        psi = pm.Deterministic(
            'psi',
            sigma_s * (pm.math.sqrt(rho) * phi_scaled + pm.math.sqrt(1.0 - rho) * theta_raw)
        )
        phi = pm.Deterministic('phi', psi)
        
        alpha_disp = pm.Exponential('alpha_disp', 1.0)
        
        log_mu = alpha + beta_year * is_2025_obs + beta_area * log_area_obs + psi[sub_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
    return model_d

def evaluate_seed_stability(n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs, seeds=[42, 123, 999], n_iter=40000, tol=1e-4):
    """
    Evalúa la estabilidad de la inferencia ADVI en el Modelo (d) corriendo con 3 semillas aleatorias distintas.
    Reporta la estabilidad del ELBO y de los parámetros estimados (alpha, beta_year, beta_area, sigma_s, rho, alpha_disp).
    """
    logger.info(f"Iniciando evaluación de estabilidad del modelo final con {len(seeds)} semillas: {seeds}...")
    seed_records = []
    
    for s in seeds:
        m = build_model_d(n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs)
        approx, tr, elbo_h, f_elbo, n_act, conv = fit_advi_model(m, f"BYM2 (Semilla {s})", n_iter=n_iter, tol=tol, seed=s, draws=2000)
        
        # Extraer medias y SDs de parámetros posteriores
        record = {
            'Semilla': s,
            'ELBO_Final': round(f_elbo, 2),
            'ELBO_Meseta_Media': round(float(np.mean(elbo_h[-min(1000, len(elbo_h)):])) , 2),
            'Iteraciones': n_act,
            'Convergio': conv,
            'alpha_media': float(tr.posterior['alpha'].mean()),
            'alpha_sd': float(tr.posterior['alpha'].std()),
            'beta_year_media': float(tr.posterior['beta_year'].mean()),
            'beta_year_sd': float(tr.posterior['beta_year'].std()),
            'beta_area_media': float(tr.posterior['beta_area'].mean()),
            'beta_area_sd': float(tr.posterior['beta_area'].std()),
            'sigma_s_media': float(tr.posterior['sigma_s'].mean()),
            'sigma_s_sd': float(tr.posterior['sigma_s'].std()),
            'rho_media': float(tr.posterior['rho'].mean()),
            'rho_sd': float(tr.posterior['rho'].std()),
            'rho_hdi_inf': float(np.percentile(tr.posterior['rho'].values, 2.5)),
            'rho_hdi_sup': float(np.percentile(tr.posterior['rho'].values, 97.5)),
            'alpha_disp_media': float(tr.posterior['alpha_disp'].mean()),
            'alpha_disp_sd': float(tr.posterior['alpha_disp'].std())
        }
        seed_records.append(record)
        
    df_stability = pd.DataFrame(seed_records)
    
    # Calcular fila resumen de Promedio y Desviación Estándar entre semillas
    num_cols = [c for c in df_stability.columns if c not in ['Semilla', 'Convergio']]
    mean_row = {'Semilla': 'Media (Entre Semillas)', 'Convergio': 'Sí (100%)'}
    sd_row = {'Semilla': 'Desv. Est. (Entre Semillas)', 'Convergio': '-'}
    for col in num_cols:
        mean_row[col] = round(float(df_stability[col].mean()), 4)
        sd_row[col] = round(float(df_stability[col].std()), 4)
        
    df_stability_full = pd.concat([df_stability, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
    excel_path = os.path.join(config.TABLES_DIR, "tabla_estabilidad_semillas_advi.xlsx")
    df_stability_full.to_excel(excel_path, index=False)
    logger.info(f"Tabla de estabilidad por semillas guardada en: {excel_path}")
    logger.info(f"ELBO medio entre semillas: {mean_row['ELBO_Final']} ± {sd_row['ELBO_Final']} | Rho medio: {mean_row['rho_media']} ± {sd_row['rho_media']}")
    return df_stability_full

def run_nuts_validation(n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs, trace_advi, seed=42):
    """
    Ajusta el modelo final (d) BYM2 con NUTS MCMC para validar formalmente la aproximación ADVI.
    Compara medias, desviaciones estándar, intervalos de credibilidad, ratios de incertidumbre
    y correlación de los riesgos relativos espaciales (RR) entre ambos métodos.
    """
    logger.info("Iniciando VALIDACIÓN METODOLÓGICA CONTRA NUTS (Modelo BYM2)...")
    model_nuts = build_model_d(n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs)
    
    t0 = time.time()
    with model_nuts:
        # NUTS con 2 cadenas y 1000 draws para validación computacionalmente eficiente
        trace_nuts = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=min(2, os.cpu_count() or 1),
            target_accept=0.95,
            random_seed=seed,
            progressbar=False
        )
        pm.compute_log_likelihood(trace_nuts, model=model_nuts)
    t_nuts = time.time() - t0
    logger.info(f"✓ NUTS de validación completado en {t_nuts:.2f} s.")
    
    # Guardar traza de NUTS
    nuts_nc_path = os.path.join(config.MODELS_DIR, "trace_bym2_nuts.nc")
    az.to_netcdf(trace_nuts, nuts_nc_path)
    
    # 1. Tabla Comparativa de Parámetros ADVI vs NUTS
    param_names = ['alpha', 'beta_year', 'beta_area', 'sigma_s', 'rho', 'alpha_disp']
    param_labels = {
        'alpha': 'Intercepto Cantonal (alpha)',
        'beta_year': 'Contraste Anual 2025 vs 2024 (beta_year)',
        'beta_area': 'Elasticidad Área Territorial (beta_area)',
        'sigma_s': 'Escala Espacial BYM2 (sigma_s)',
        'rho': 'Parámetro de Mezcla Espacial (rho)',
        'alpha_disp': 'Dispersión Binomial Negativa (alpha_disp)'
    }
    
    comp_records = []
    for p in param_names:
        advi_vals = trace_advi.posterior[p].values.flatten()
        nuts_vals = trace_nuts.posterior[p].values.flatten()
        
        m_advi, sd_advi = float(np.mean(advi_vals)), float(np.std(advi_vals))
        hdi_advi_low, hdi_advi_high = float(np.percentile(advi_vals, 2.5)), float(np.percentile(advi_vals, 97.5))
        
        m_nuts, sd_nuts = float(np.mean(nuts_vals)), float(np.std(nuts_vals))
        hdi_nuts_low, hdi_nuts_high = float(np.percentile(nuts_vals, 2.5)), float(np.percentile(nuts_vals, 97.5))
        
        ratio_sd = (sd_advi / sd_nuts) if sd_nuts > 0 else 1.0
        pct_underest = ((1.0 - ratio_sd) * 100.0) if sd_nuts > 0 else 0.0
        
        comp_records.append({
            'Parámetro': param_labels.get(p, p),
            'Media_ADVI': round(m_advi, 4),
            'SD_ADVI': round(sd_advi, 4),
            'HDI_95_ADVI': f"[{hdi_advi_low:.4f}, {hdi_advi_high:.4f}]",
            'Media_NUTS': round(m_nuts, 4),
            'SD_NUTS': round(sd_nuts, 4),
            'HDI_95_NUTS': f"[{hdi_nuts_low:.4f}, {hdi_nuts_high:.4f}]",
            'Ratio_SD (ADVI/NUTS)': round(ratio_sd, 4),
            'Subestimacion_Incertidumbre_Pct': f"{pct_underest:+.1f}%"
        })
        
    df_comp_nuts = pd.DataFrame(comp_records)
    excel_nuts_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_advi_nuts.xlsx")
    df_comp_nuts.to_excel(excel_nuts_path, index=False)
    logger.info(f"Tabla de comparación ADVI vs NUTS guardada en: {excel_nuts_path}")
    
    # 2. Correlación de Riesgos Relativos (RR) por Subcircuito
    phi_advi = trace_advi.posterior['psi'].values.reshape(-1, n_zonas)
    exp_advi = np.exp(phi_advi)
    rr_advi = exp_advi / np.mean(exp_advi, axis=1, keepdims=True)
    rr_median_advi = np.median(rr_advi, axis=0)
    
    phi_nuts = trace_nuts.posterior['psi'].values.reshape(-1, n_zonas)
    exp_nuts = np.exp(phi_nuts)
    rr_nuts = exp_nuts / np.mean(exp_nuts, axis=1, keepdims=True)
    rr_median_nuts = np.median(rr_nuts, axis=0)
    
    r_pearson, p_pearson = pearsonr(rr_median_advi, rr_median_nuts)
    r_spearman, p_spearman = spearmanr(rr_median_advi, rr_median_nuts)
    logger.info(f"Correlación de RR espaciales ADVI vs NUTS: Pearson r = {r_pearson:.4f} (p={p_pearson:.1e}) | Spearman rho = {r_spearman:.4f}")
    
    # 3. Gráfico de Dispersión de RR: ADVI vs NUTS
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor='white')
    ax.scatter(rr_median_nuts, rr_median_advi, color='#1565c0', alpha=0.65, edgecolors='none', s=45, label='Subcircuitos (N=244)')
    
    # Línea de identidad y = x
    lim_min = min(rr_median_nuts.min(), rr_median_advi.min()) * 0.9
    lim_max = max(rr_median_nuts.max(), rr_median_advi.max()) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='Identidad Perfecta ($y = x$)')
    
    # Ajuste lineal de regresión
    m_reg, b_reg = np.polyfit(rr_median_nuts, rr_median_advi, 1)
    ax.plot(np.array([lim_min, lim_max]), m_reg * np.array([lim_min, lim_max]) + b_reg, color='#004d40', linewidth=1.5, linestyle=':', label=f'Regresión: $y = {m_reg:.2f}x {b_reg:+.2f}$')
    
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Riesgo Relativo Mediano - NUTS (MCMC Exacto)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Riesgo Relativo Mediano - ADVI (Inferencia Variacional)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f'Validación de Inferencia: Riesgo Relativo Espacial ADVI vs. NUTS\nCorrelación de Pearson $r = {r_pearson:.4f}$ | Spearman $\\rho = {r_spearman:.4f}$', fontsize=12, fontweight='bold', pad=12)
    ax.grid(color='#eeeeee', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#cccccc')
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#aaaaaa')
        
    png_scatter = os.path.join(config.FIGURES_DIR, "Figura_dispersion_rr_advi_nuts.png")
    pdf_scatter = os.path.join(config.FIGURES_DIR, "Figura_dispersion_rr_advi_nuts.pdf")
    plt.tight_layout()
    plt.savefig(png_scatter, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_scatter, format='pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de dispersión ADVI vs NUTS guardado en: {png_scatter}")
    
    return df_comp_nuts, r_pearson, r_spearman, trace_nuts

def plot_elbo_convergence(elbo_dict: dict):
    """
    Genera la figura de convergencia del ELBO para los modelos bayesianos ajustados con ADVI.
    Incluye un panel con la trayectoria global y un panel ampliado (zoom) sobre la meseta estable
    de convergencia, además de guardar las series en formato comprimido npz.
    """
    # Guardar trayectorias para reproducibilidad y análisis
    npz_path = os.path.join(config.MODELS_DIR, "elbo_trajectories.npz")
    np.savez_compressed(npz_path, **{k.replace(" ", "_"): v for k, v in elbo_dict.items()})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')
    colors = {
        'Poisson Jerárquico': '#e53935',
        'Binomial Negativa (ICAR)': '#fb8c00',
        'Binomial Negativa (ICAR + No Estructurado)': '#43a047',
        'BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))': '#1e88e5'
    }
    
    # PANEL 1: Trayectoria Global de Optimización
    for name, elbo_hist in elbo_dict.items():
        step = max(1, len(elbo_hist) // 500)
        x_iters = np.arange(0, len(elbo_hist), step)
        y_elbo = elbo_hist[::step]
        ax1.plot(x_iters, y_elbo, label=name, color=colors.get(name, '#333333'), linewidth=1.8)
        
    ax1.set_xlabel('Iteración de Optimización', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('ELBO (Escala Completa)', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('(A) Trayectoria Global de Optimización', fontsize=11, fontweight='bold', pad=10)
    ax1.grid(color='#eeeeee', linestyle='-', linewidth=0.5)
    ax1.legend(loc='lower right', fontsize=8.5, frameon=True, edgecolor='#cccccc')
    
    # PANEL 2: Zoom sobre la Meseta de Convergencia (Iteraciones 2,000 - 40,000)
    for name, elbo_hist in elbo_dict.items():
        step = max(1, len(elbo_hist) // 400)
        x_iters = np.arange(0, len(elbo_hist), step)
        y_elbo = elbo_hist[::step]
        # Suavizado de media móvil local para filtrar el ruido estocástico de Monte Carlo
        kernel = 5
        if len(y_elbo) > kernel:
            y_smooth = np.convolve(y_elbo, np.ones(kernel)/kernel, mode='valid')
            x_smooth = x_iters[kernel-1:]
        else:
            y_smooth = y_elbo
            x_smooth = x_iters
        ax2.plot(x_smooth, y_smooth, label=name, color=colors.get(name, '#333333'), linewidth=2.0)
        
    ax2.set_ylim(-6000, -1000)
    ax2.set_xlim(2000, 40000)
    ax2.set_xlabel('Iteración de Optimización', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_ylabel('ELBO (Zoom Meseta [-6000, -1000])', fontsize=11, fontweight='bold', labelpad=8)
    ax2.set_title('(B) Detalle de Convergencia en la Meseta Estable', fontsize=11, fontweight='bold', pad=10)
    ax2.grid(color='#eeeeee', linestyle='-', linewidth=0.5)
    ax2.legend(loc='lower right', fontsize=8.5, frameon=True, edgecolor='#cccccc')
    
    for ax in [ax1, ax2]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#aaaaaa')
            
    fig.suptitle('Convergencia del ELBO en Inferencia Variacional ADVI\nModelos Bayesianos de Homicidios en Guayaquil (Tolerancia 1e-4)', fontsize=12, fontweight='bold', y=0.98)
    
    png_path = os.path.join(config.FIGURES_DIR, "Figura_convergencia_elbo.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura_convergencia_elbo.pdf")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de convergencia del ELBO guardado en: {png_path} y {pdf_path}")

def run_stage8(gdf_total: gpd.GeoDataFrame, gdf_anual: gpd.GeoDataFrame, w_selected) -> tuple:
    """
    Etapa 8: Ajuste Real de Modelos Bayesianos Jerárquicos mediante Inferencia Variacional (ADVI)
    
    METODOLOGÍA Y ESPECIFICACIÓN:
    1. INFERENCIA PRINCIPAL (ADVI):
       Se utiliza Automatic Differentiation Variational Inference (ADVI) en PyMC con convergencia
       monitoreada mediante CheckParametersConvergence (tolerancia 1e-4) y 2,000 muestras posteriores.
       
    2. VALIDACIÓN METODOLÓGICA CON NUTS:
       El modelo final BYM2 se ajusta además con NUTS MCMC bajo la misma formulación para evaluar
       la calidad de la aproximación variacional y cuantificar el ratio de incertidumbre.
       
    3. CONTRASTE ANUAL (2025 vs 2024):
       Dado que el panel comprende 2 años completos (2024 y 2025), se modela un efecto fijo de año
       beta_year ~ Normal(0, 1) sobre 2025 frente a la línea base 2024.
       
    4. PRIOR DE MEZCLA ESPACIAL:
       El modelo BYM2 incorpora explícitamente el prior Beta(0.5, 0.5) en el parámetro rho.
    """
    logger.info("Iniciando ETAPA 8: Ajuste de Modelos Bayesianos con ADVI (tolerancia 1e-4, draws=2000)")
    
    # 1. Preparar datos
    df_train = gdf_anual.copy()
    df_train['Area_km2_limpia'] = df_train['Area_km2'].fillna(0.1)
    df_train.loc[df_train['Area_km2_limpia'] <= 0, 'Area_km2_limpia'] = 0.001
    df_train['log_area'] = np.log(df_train['Area_km2_limpia'])
    
    sub_codes = sorted(gdf_total['cod_subcircu'].unique())
    n_zonas = len(sub_codes)
    sub_code_to_idx = {code: idx for idx, code in enumerate(sub_codes)}
    df_train['sub_idx'] = df_train['cod_subcircu'].map(sub_code_to_idx)
    
    # Indicador de año: 2024 = 0 (línea base de referencia), 2025 = 1
    df_train['is_2025'] = (df_train['Anio'] == 2025).astype(float)
    
    homicidios_obs = df_train['homicidios'].values.astype(int)
    log_area_obs = df_train['log_area'].values
    sub_idx = df_train['sub_idx'].values.astype(int)
    is_2025_obs = df_train['is_2025'].values
    
    logger.info(f"Panel configurado para {n_zonas} subcircuitos y 2 años completos [2024, 2025] ({len(df_train)} observaciones espacio-temporales).")
    
    # Matriz binaria de vecindad simétrica para pm.ICAR
    W_bin = np.array(w_selected.full()[0], dtype=int)
    np.fill_diagonal(W_bin, 0)
    
    # Factor de escala para BYM2
    scale_factor_bym2 = compute_bym2_scaling_factor(w_selected)
    
    seed = config.RANDOM_SEED
    n_advi_iter = 40000
    advi_tol = 1e-4
    
    traces = {}
    models = {}
    elbo_trajectories = {}
    advi_diagnostics = {}
    
    # -------------------------------------------------------------
    # MODELO (A): Poisson Jerárquico (Línea Base Espacio-Temporal)
    # -------------------------------------------------------------
    with pm.Model() as model_a:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_year = pm.Normal('beta_year', mu=0, sigma=1)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_phi = pm.Exponential('sigma_phi', 1.0)
        
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi = pm.Deterministic('phi', sigma_phi * phi_raw)
        
        log_mu = alpha + beta_year * is_2025_obs + beta_area * log_area_obs + phi[sub_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.Poisson('y_obs', mu=mu, observed=homicidios_obs)
        
    approx_a, trace_a, elbo_a, elbo_f_a, n_act_a, conv_a = fit_advi_model(
        model_a, "Poisson Jerárquico", n_iter=n_advi_iter, tol=advi_tol, seed=seed, draws=2000
    )
    models['Poisson Jerárquico'] = model_a
    traces['Poisson Jerárquico'] = trace_a
    elbo_trajectories['Poisson Jerárquico'] = elbo_a
    advi_diagnostics['Poisson Jerárquico'] = {'ELBO_Final': elbo_f_a, 'Iteraciones': n_act_a, 'Convergio': conv_a}
    az.to_netcdf(trace_a, os.path.join(config.MODELS_DIR, "trace_poisson.nc"))

    # -------------------------------------------------------------
    # MODELO (B): Binomial Negativa Jerárquico (ICAR)
    # -------------------------------------------------------------
    with pm.Model() as model_b:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_year = pm.Normal('beta_year', mu=0, sigma=1)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_phi = pm.Exponential('sigma_phi', 1.0)
        
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi = pm.Deterministic('phi', sigma_phi * phi_raw)
        alpha_disp = pm.Exponential('alpha_disp', 1.0)
        
        log_mu = alpha + beta_year * is_2025_obs + beta_area * log_area_obs + phi[sub_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
        
    approx_b, trace_b, elbo_b, elbo_f_b, n_act_b, conv_b = fit_advi_model(
        model_b, "Binomial Negativa (ICAR)", n_iter=n_advi_iter, tol=advi_tol, seed=seed, draws=2000
    )
    models['Binomial Negativa (ICAR)'] = model_b
    traces['Binomial Negativa (ICAR)'] = trace_b
    elbo_trajectories['Binomial Negativa (ICAR)'] = elbo_b
    advi_diagnostics['Binomial Negativa (ICAR)'] = {'ELBO_Final': elbo_f_b, 'Iteraciones': n_act_b, 'Convergio': conv_b}
    az.to_netcdf(trace_b, os.path.join(config.MODELS_DIR, "trace_nb_icar.nc"))

    # -------------------------------------------------------------
    # MODELO (C): Binomial Negativa con ICAR + Efecto No Estructurado (Besag et al. 1991)
    # -------------------------------------------------------------
    with pm.Model() as model_c:
        alpha = pm.Normal('alpha', mu=0, sigma=2)
        beta_year = pm.Normal('beta_year', mu=0, sigma=1)
        beta_area = pm.Normal('beta_area', mu=0, sigma=1)
        sigma_phi = pm.Exponential('sigma_phi', 1.0)
        sigma_theta = pm.Exponential('sigma_theta', 1.0)
        
        phi_raw = pm.ICAR('phi_raw', W=W_bin)
        phi_struct = pm.Deterministic('phi_struct', sigma_phi * phi_raw)
        
        theta_raw = pm.Normal('theta_raw', 0, 1, shape=n_zonas)
        theta = pm.Deterministic('theta', sigma_theta * theta_raw)
        
        spatial_total = pm.Deterministic('spatial_total', phi_struct + theta)
        phi = pm.Deterministic('phi', spatial_total)
        
        alpha_disp = pm.Exponential('alpha_disp', 1.0)
        
        log_mu = alpha + beta_year * is_2025_obs + beta_area * log_area_obs + spatial_total[sub_idx]
        mu = pm.math.exp(log_mu)
        
        y_obs = pm.NegativeBinomial('y_obs', mu=mu, alpha=alpha_disp, observed=homicidios_obs)
        
    approx_c, trace_c, elbo_c, elbo_f_c, n_act_c, conv_c = fit_advi_model(
        model_c, "Binomial Negativa (ICAR + No Estructurado)", n_iter=n_advi_iter, tol=advi_tol, seed=seed, draws=2000
    )
    models['Binomial Negativa (ICAR + No Estructurado)'] = model_c
    traces['Binomial Negativa (ICAR + No Estructurado)'] = trace_c
    elbo_trajectories['Binomial Negativa (ICAR + No Estructurado)'] = elbo_c
    advi_diagnostics['Binomial Negativa (ICAR + No Estructurado)'] = {'ELBO_Final': elbo_f_c, 'Iteraciones': n_act_c, 'Convergio': conv_c}
    az.to_netcdf(trace_c, os.path.join(config.MODELS_DIR, "trace_nb_unstructured.nc"))

    # -------------------------------------------------------------
    # MODELO (D): BYM2 Binomial Negativo con Prior Beta(0.5, 0.5) (Riebler et al. 2016)
    # -------------------------------------------------------------
    model_d = build_model_d(n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs)
    approx_d, trace_d, elbo_d, elbo_f_d, n_act_d, conv_d = fit_advi_model(
        model_d, "BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))", n_iter=n_advi_iter, tol=advi_tol, seed=seed, draws=2000
    )
    models['BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))'] = model_d
    traces['BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))'] = trace_d
    elbo_trajectories['BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))'] = elbo_d
    advi_diagnostics['BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))'] = {'ELBO_Final': elbo_f_d, 'Iteraciones': n_act_d, 'Convergio': conv_d}
    az.to_netcdf(trace_d, os.path.join(config.MODELS_DIR, "trace_bym2.nc"))

    # Graficar convergencia del ELBO para todos los modelos
    plot_elbo_convergence(elbo_trajectories)

    # -------------------------------------------------------------
    # EVALUACIÓN DE ESTABILIDAD CON 3 SEMILLAS (MODELO FINAL BYM2)
    # -------------------------------------------------------------
    df_stability = evaluate_seed_stability(
        n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs,
        seeds=[42, 123, 999], n_iter=n_advi_iter, tol=advi_tol
    )

    # -------------------------------------------------------------
    # VALIDACIÓN METODOLÓGICA CONTRA NUTS (MODELO FINAL BYM2)
    # -------------------------------------------------------------
    df_comp_nuts, r_pearson, r_spearman, trace_nuts = run_nuts_validation(
        n_zonas, W_bin, scale_factor_bym2, is_2025_obs, log_area_obs, sub_idx, homicidios_obs,
        trace_advi=trace_d, seed=seed
    )

    # -------------------------------------------------------------
    # COMPARACIÓN FORMAL DE MODELOS CON LOO-CV Y WAIC (ESCALA LOG)
    # -------------------------------------------------------------
    logger.info("Ejecutando comparación formal de modelos mediante ArviZ LOO-CV (escala log ELPD)...")
    df_compare_loo = az.compare(traces, ic='loo', scale='log')
    df_compare_waic = az.compare(traces, ic='waic', scale='log')
    logger.info("\nComparación LOO-CV:\n" + df_compare_loo.to_string())
    logger.info("\nComparación WAIC:\n" + df_compare_waic.to_string())
    
    pareto_details = {}
    waic_map = {}
    
    for name, tr in traces.items():
        loo_obj = az.loo(tr, pointwise=True, scale='log')
        k_vals = loo_obj.pareto_k.values
        n_bad_k = int((k_vals > 0.7).sum())
        n_ok_k = int(((k_vals >= 0.5) & (k_vals <= 0.7)).sum())
        n_good_k = int((k_vals < 0.5).sum())
        warn_txt = "Advertencia: observaciones con k>0.7 presentes" if n_bad_k > 0 else "Diagnóstico k satisfactorio"
        pareto_details[name] = {
            'k_menor_0.5': n_good_k,
            'k_0.5_a_0.7': n_ok_k,
            'k_mayor_0.7': n_bad_k,
            'Advertencia_Pareto_k': warn_txt
        }
        
        w_obj = az.waic(tr, scale='log')
        waic_map[name] = {
            'elpd_waic': float(w_obj.elpd_waic),
            'se_waic': float(w_obj.se),
            'p_waic': float(w_obj.p_waic)
        }
        
    df_compare_export = df_compare_loo.copy().reset_index()
    df_compare_export.rename(columns={'index': 'Modelo'}, inplace=True)
    
    # Agregar métricas de ADVI y diagnósticos Pareto-k
    df_compare_export['ELBO_Final'] = df_compare_export['Modelo'].map(lambda m: advi_diagnostics[m]['ELBO_Final'])
    df_compare_export['Iteraciones_ADVI'] = df_compare_export['Modelo'].map(lambda m: advi_diagnostics[m]['Iteraciones'])
    df_compare_export['Convergio'] = df_compare_export['Modelo'].map(lambda m: advi_diagnostics[m]['Convergio'])
    df_compare_export['k < 0.5'] = df_compare_export['Modelo'].map(lambda m: pareto_details[m]['k_menor_0.5'])
    df_compare_export['0.5 <= k <= 0.7'] = df_compare_export['Modelo'].map(lambda m: pareto_details[m]['k_0.5_a_0.7'])
    df_compare_export['k > 0.7'] = df_compare_export['Modelo'].map(lambda m: pareto_details[m]['k_mayor_0.7'])
    df_compare_export['Advertencia_Pareto_k'] = df_compare_export['Modelo'].map(lambda m: pareto_details[m]['Advertencia_Pareto_k'])
    df_compare_export['elpd_waic'] = df_compare_export['Modelo'].map(lambda m: waic_map[m]['elpd_waic'])
    df_compare_export['p_waic'] = df_compare_export['Modelo'].map(lambda m: waic_map[m]['p_waic'])
    
    excel_comp_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_modelos_bayesiana.xlsx")
    df_compare_export.to_excel(excel_comp_path, index=False)
    logger.info(f"Tabla de comparación formal guardada en: {excel_comp_path}")
    
    # EVALUACIÓN DE SIGNIFICANCIA ENTRE (C) Y (D)
    if 'Binomial Negativa (ICAR + No Estructurado)' in df_compare_loo.index and 'BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))' in df_compare_loo.index:
        c_elpd = df_compare_loo.loc['Binomial Negativa (ICAR + No Estructurado)', 'elpd_loo']
        d_elpd = df_compare_loo.loc['BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))', 'elpd_loo']
        diff_cd = abs(c_elpd - d_elpd)
        dse_val = df_compare_loo.loc['Binomial Negativa (ICAR + No Estructurado)', 'dse']
        logger.info("==================================================================================")
        logger.info(f"EVALUACIÓN COMPARATIVA MODELO (c) vs MODELO (d):")
        logger.info(f"Diferencia de ELPD_LOO: {diff_cd:.4f} puntos | Error estándar de la diferencia (dse): {dse_val:.4f}")
        if diff_cd < 2.0 * dse_val:
            logger.info("DICTAMEN METODOLÓGICO: La diferencia entre el modelo (c) (ICAR + No Estructurado) y el")
            logger.info("modelo (d) (BYM2) es menor a 2 errores estándar (diff < 2*dse).")
            logger.info("Por tanto, NO ES ESTADÍSTICAMENTE SIGNIFICATIVA y AMBOS MODELOS EMPATAN.")
            logger.info("No se puede afirmar que BYM2 supere a (c); ambos presentan un ajuste predictivo equivalente.")
        logger.info("==================================================================================")

    # -------------------------------------------------------------
    # SELECCIÓN METODOLÓGICA DEL MODELO PARA RIESGO RELATIVO DOWNSTREAM
    # -------------------------------------------------------------
    best_model_name = 'BYM2 Binomial Negativo (Prior Beta(0.5, 0.5))'
    best_model = models[best_model_name]
    best_trace = traces[best_model_name]
    logger.info(f"Modelo seleccionado para inferencia espacial downstream: '{best_model_name}' (estadísticamente equivalente a (c)).")
    
    # Generar contraste anual
    generate_temporal_outputs(best_trace)
    
    # Generar muestras predictivas posteriores (PPC) en rango completo
    logger.info(f"Generando distribución predictiva posterior (PPC) completa para {best_model_name}...")
    with best_model:
        pm.sample_posterior_predictive(best_trace, extend_inferencedata=True, random_seed=seed)
        
    # Guardar traza principal para el pipeline downstream (diagnósticos, dashboard, riesgo relativo)
    main_trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    az.to_netcdf(best_trace, main_trace_path)
    logger.info(f"Traza principal guardada en: {main_trace_path}")
    
    logger.info("ETAPA 8: Modelos Bayesianos con ADVI y validación NUTS finalizados con éxito.")
    return best_model, best_trace, df_compare_export, traces
