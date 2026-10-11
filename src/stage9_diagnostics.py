import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage9")

def save_diagnostic_plot(name: str):
    """Guarda el gráfico actual de diagnóstico en alta resolución (PNG y PDF)"""
    png_path = os.path.join(config.FIGURES_DIR, f"{name}.png")
    pdf_path = os.path.join(config.FIGURES_DIR, f"{name}.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, format='pdf')
    plt.close()
    logger.info(f"Gráfico de diagnóstico guardado: {png_path} y {pdf_path}")

def run_stage9(trace) -> dict:
    """
    Etapa 9: Diagnósticos de Convergencia Bayesiana y Verificación Posterior
    - Calcula resumen posterior formal ArviZ: r_hat, bulk-ESS, tail-ESS y divergencias reales de NUTS.
    - Genera ArviZ traceplots, posterior plots, forest plots, energy plot y autocorrelogramas.
    - Realiza Posterior Predictive Checks (PPC) sobre el rango completo sin truncamiento artificial.
    - Calcula criterios de información predictiva WAIC y LOO en escala logarítmica (ELPD) de forma consistente.
    - Sin resultados fabricados ni simulaciones artificiales.
    """
    logger.info("Iniciando ETAPA 9: Diagnósticos Bayesianos Reales (NUTS)")
    
    # 1. Identificar variables clave presentes en la traza
    candidate_vars = ['alpha', 'beta_area', 'sigma_phi', 'sigma_theta', 'sigma_s', 'rho', 'sigma_t', 'alpha_disp']
    model_vars = [v for v in candidate_vars if v in trace.posterior]
    logger.info(f"Parámetros clave identificados para diagnósticos: {model_vars}")
    
    # 2. Resumen Estadístico Completo (ArviZ summary)
    logger.info("Calculando resumen posterior ArviZ (r_hat, ESS bulk y ESS tail)...")
    summary = az.summary(trace, var_names=model_vars, round_to=4)
    logger.info("\n" + summary.to_string())
    
    # Extraer conteo de divergencias NUTS
    n_divergences = 0
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        n_divergences = int(trace.sample_stats.diverging.values.sum())
    logger.info(f"Divergencias NUTS registradas en el muestreo: {n_divergences}")
    
    # Agregar fila de diagnóstico general al resumen
    summary_export = summary.copy()
    
    # Guardar resumen en Excel
    excel_summary_path = os.path.join(config.TABLES_DIR, "tabla_diagnosticos_bayesianos.xlsx")
    with pd.ExcelWriter(excel_summary_path) as writer:
        summary_export.to_excel(writer, sheet_name="Parametros_Posteriores")
        pd.DataFrame([
            {"Diagnóstico": "Total Cadenas", "Valor": trace.posterior.dims.get('chain', 4)},
            {"Diagnóstico": "Muestras por Cadena (Draws)", "Valor": trace.posterior.dims.get('draw', 1000)},
            {"Diagnóstico": "Divergencias NUTS", "Valor": n_divergences},
            {"Diagnóstico": "R_hat Máximo", "Valor": float(summary['r_hat'].max())},
            {"Diagnóstico": "ESS Bulk Mínimo", "Valor": float(summary['ess_bulk'].min())},
            {"Diagnóstico": "ESS Tail Mínimo", "Valor": float(summary['ess_tail'].min())}
        ]).to_excel(writer, sheet_name="Diagnosticos_Convergencia", index=False)
        
    logger.info(f"Resumen de diagnósticos y convergencia guardado en: {excel_summary_path}")
    
    # 3. Generación de Gráficos de Diagnóstico
    # Gráfico 14: Traceplots
    logger.info("Generando traceplots...")
    az.plot_trace(trace, var_names=model_vars)
    save_diagnostic_plot("Figura14_bayesian_traceplots")
    
    # Gráfico 15: Posterior distributions
    logger.info("Generando gráficos de densidad posterior...")
    plot_post_vars = [v for v in ['alpha', 'beta_area', 'sigma_phi', 'sigma_s', 'alpha_disp'] if v in trace.posterior]
    az.plot_posterior(trace, var_names=plot_post_vars)
    save_diagnostic_plot("Figura15_bayesian_posterior")
    
    # Gráfico 16: Forest plots (Intervalos HDI)
    logger.info("Generando forest plot...")
    forest_vars = [v for v in ['beta_area', 'sigma_phi', 'sigma_theta', 'sigma_s', 'sigma_t'] if v in trace.posterior]
    if forest_vars:
        az.plot_forest(trace, var_names=forest_vars, combined=True)
        save_diagnostic_plot("Figura16_bayesian_forest")
    
    # Gráfico 17: Energy plot real (convergencia NUTS)
    logger.info("Generando energy plot a partir de sample_stats...")
    try:
        if hasattr(trace, 'sample_stats') and 'energy' in trace.sample_stats:
            az.plot_energy(trace)
            save_diagnostic_plot("Figura17_bayesian_energy")
        else:
            logger.warning("sample_stats.energy no está presente en la traza; no se genera gráfico sintético.")
    except Exception as e:
        logger.error(f"Error al graficar energy plot: {e}. No se sustituye por datos simulados.")
    
    # Gráfico 18: Autocorrelación para parámetros clave
    logger.info("Generando gráfico de autocorrelación...")
    autocorr_vars = [v for v in ['alpha', 'beta_area', 'sigma_phi', 'sigma_s'] if v in trace.posterior][:2]
    az.plot_autocorr(trace, var_names=autocorr_vars, max_lag=25)
    save_diagnostic_plot("Figura18_bayesian_autocorrelation")
    
    # Gráfico 19: Posterior Predictive Checks (PPC) en rango completo
    logger.info("Generando Posterior Predictive Check (PPC) sobre el rango completo...")
    if hasattr(trace, 'posterior_predictive') and 'y_obs' in trace.posterior_predictive:
        try:
            fig, ax = plt.subplots(figsize=(9, 5.5))
            obs = trace.observed_data['y_obs'].values.flatten()
            pred_raw = trace.posterior_predictive['y_obs'].values
            pred = pred_raw.reshape(-1, len(obs))
            
            # Submuestrear 30 curvas predictivas
            np.random.seed(config.RANDOM_SEED)
            idx_sims = np.random.choice(len(pred), min(30, len(pred)), replace=False)
            
            # Graficar la primera para la leyenda
            sns.kdeplot(pred[idx_sims[0]], color='#4a90e2', alpha=0.3, linewidth=0.8, clip=(0, None), ax=ax, label='Réplicas predictivas posteriores ($y_{rep}$)')
            for idx in idx_sims[1:]:
                sns.kdeplot(pred[idx], color='#4a90e2', alpha=0.3, linewidth=0.8, clip=(0, None), ax=ax)
                
            # Curva observada
            sns.kdeplot(obs, color='#111111', linewidth=2.5, clip=(0, None), ax=ax, label=f'Datos observados (y, max={obs.max()})')
            
            # Media predictiva posterior
            mean_pred = pred.mean(axis=0)
            sns.kdeplot(mean_pred, color='#e65100', linewidth=2.2, linestyle='--', clip=(0, None), ax=ax, label='Media predictiva posterior')
            
            # Rango completo sin recorte artificial
            max_x = max(float(obs.max()), float(np.percentile(pred, 99.5))) * 1.05
            ax.set_xlim(0, max_x)
            ax.set_title("Figura 19: Comprobación Predictiva Posterior (PPC) - Rango Completo", fontsize=12, fontweight='bold')
            ax.set_xlabel("Número de Homicidios por Subcircuito", fontsize=10, labelpad=8)
            ax.set_ylabel("Densidad de Probabilidad", fontsize=10, labelpad=8)
            ax.legend(loc='upper right', frameon=True, facecolor='#f8f9fa')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            save_diagnostic_plot("Figura19_bayesian_ppc")
        except Exception as e:
            logger.error(f"Error al generar el gráfico de PPC: {e}")
    else:
        logger.warning("No se encontraron muestras en trace.posterior_predictive para generar PPC.")
        
    # 4. Criterios de Información (WAIC y LOO) en Escala Log (ELPD)
    logger.info("Calculando criterios WAIC y LOO en escala logarítmica (ELPD)...")
    metrics_info = {}
    
    waic_res = az.waic(trace, scale="log")
    loo_res = az.loo(trace, scale="log")
    
    metrics_info["ELPD_WAIC"] = float(waic_res.elpd_waic)
    metrics_info["ELPD_WAIC_SE"] = float(waic_res.se)
    metrics_info["p_WAIC"] = float(waic_res.p_waic)
    metrics_info["ELPD_LOO"] = float(loo_res.elpd_loo)
    metrics_info["ELPD_LOO_SE"] = float(loo_res.se)
    metrics_info["p_LOO"] = float(loo_res.p_loo)
    metrics_info["Divergencias_NUTS"] = n_divergences
    metrics_info["R_hat_max"] = float(summary['r_hat'].max())
    metrics_info["ESS_bulk_min"] = float(summary['ess_bulk'].min())
    metrics_info["ESS_tail_min"] = float(summary['ess_tail'].min())
    
    logger.info(f"ELPD_WAIC: {waic_res.elpd_waic:.4f} (se: {waic_res.se:.4f}, p_waic: {waic_res.p_waic:.2f})")
    logger.info(f"ELPD_LOO: {loo_res.elpd_loo:.4f} (se: {loo_res.se:.4f}, p_loo: {loo_res.p_loo:.2f})")
    
    # Guardar criterios en excel complementario
    df_metrics = pd.DataFrame(list(metrics_info.items()), columns=["Métrica", "Valor"])
    df_metrics.to_excel(os.path.join(config.TABLES_DIR, "tabla_criterios_informacion.xlsx"), index=False)
    
    logger.info("ETAPA 9: Diagnósticos Bayesianos finalizada con éxito.")
    return metrics_info
