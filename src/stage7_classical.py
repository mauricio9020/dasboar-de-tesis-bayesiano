import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage7")

def run_stage7(df_anual: pd.DataFrame) -> tuple:
    """
    Etapa 7: Modelos Clásicos
    - Ajusta un modelo Poisson frecuentista con offset de área.
    - Ajusta un modelo Binomial Negativo (NB2) con offset de área.
    - Compara ambos modelos usando AIC, BIC, Log-Likelihood y Pseudo R².
    - Analiza residuos y genera gráfico Q-Q de diagnóstico numerado a 300 dpi.
    - Guarda la tabla de comparación de modelos.
    """
    logger.info("Iniciando ETAPA 7: Ajuste de Modelos Clásicos (Frecuentistas)")
    
    # Preparar datos: asegurar que no haya áreas cero o negativas
    df_model = df_anual.copy()
    df_model['Area_km2_limpia'] = df_model['Area_km2'].fillna(0.1)
    df_model.loc[df_model['Area_km2_limpia'] <= 0, 'Area_km2_limpia'] = 0.001
    df_model['log_area'] = np.log(df_model['Area_km2_limpia'])
    
    # Asegurar que Anio sea categórico para meter efectos fijos temporales
    df_model['Anio_Cat'] = df_model['Anio'].astype(str)
    
    # 1. Ajustar Modelo Poisson con log(área) como covariable estimada
    logger.info("Ajustando modelo de regresión Poisson (log(área) como covariable)...")
    poisson_mod = smf.poisson('homicidios ~ C(Anio_Cat) + log_area', data=df_model)
    poisson_res = poisson_mod.fit(disp=0)
    
    # 2. Ajustar Modelo Binomial Negativo (NB2) con log(área) como covariable estimada
    logger.info("Ajustando modelo de regresión Binomial Negativo (NB2)...")
    try:
        nb_mod = smf.negativebinomial('homicidios ~ C(Anio_Cat) + log_area', data=df_model)
        nb_res = nb_mod.fit(disp=0)
    except Exception as e:
        logger.warning(f"Error al ajustar Binomial Negativa con formula estándar: {e}. Reintentando con GLM...")
        X_vars = sm.add_constant(pd.concat([
            pd.get_dummies(df_model['Anio_Cat'], drop_first=True, dtype=float),
            df_model['log_area']
        ], axis=1))
        nb_res = sm.GLM(
            df_model['homicidios'], 
            X_vars,
            family=sm.families.NegativeBinomial()
        ).fit()
    
    # 3. Comparación de Métricas
    # Extraer métricas para Poisson
    pois_ll = poisson_res.llf
    pois_aic = poisson_res.aic
    pois_bic = poisson_res.bic
    # Pseudo R² de McFadden
    pois_pr2 = 1.0 - (poisson_res.llf / poisson_res.llnull)
    
    # Extraer métricas para Binomial Negativo (si se ajustó con negativebinomial)
    if hasattr(nb_res, 'aic'):
        nb_ll = nb_res.llf
        nb_aic = nb_res.aic
        nb_bic = nb_res.bic
        nb_pr2 = 1.0 - (nb_res.llf / nb_res.llnull)
    else:
        # GLM no tiene llf o aic a veces
        nb_ll = nb_res.llf
        nb_aic = nb_res.aic
        nb_bic = nb_res.bic
        nb_pr2 = np.nan # no disponible fácilmente para GLM
        
    df_compare = pd.DataFrame({
        "Métrica": ["Log-Likelihood", "AIC (Akaike)", "BIC (Bayesiano)", "Pseudo R² (McFadden)"],
        "Modelo Poisson": [pois_ll, pois_aic, pois_bic, pois_pr2],
        "Modelo Binomial Negativo": [nb_ll, nb_aic, nb_bic, nb_pr2]
    })
    
    # Redondear valores
    for col in ["Modelo Poisson", "Modelo Binomial Negativo"]:
        df_compare[col] = df_compare[col].round(4)
        
    logger.info("Comparación de modelos clásicos finalizada:")
    logger.info("\n" + df_compare.to_string())
    
    # 4. Gráfico 13: Q-Q Plot de Residuos
    plt.figure(figsize=(10, 5))
    
    # Extraer residuos de Pearson/Devianza
    pois_resid = poisson_res.resid_pearson
    if hasattr(nb_res, 'resid_pearson'):
        nb_resid = nb_res.resid_pearson
    else:
        nb_resid = nb_res.resid_response # fall-back
        
    # Graficar lado a lado
    ax1 = plt.subplot(1, 2, 1)
    stats.probplot(pois_resid, dist="norm", plot=ax1)
    ax1.set_title("A: Q-Q Plot Residuos Poisson")
    ax1.set_xlabel("Cuantiles Teóricos")
    ax1.set_ylabel("Residuos de Pearson")
    ax1.grid(True)
    
    ax2 = plt.subplot(1, 2, 2)
    stats.probplot(nb_resid, dist="norm", plot=ax2)
    ax2.set_title("B: Q-Q Plot Residuos Binomial Negativo")
    ax2.set_xlabel("Cuantiles Teóricos")
    ax2.set_ylabel("Residuos de Pearson")
    ax2.grid(True)
    
    plt.suptitle("Figura 13: Diagnóstico de Residuos de Modelos Clásicos")
    
    # Guardar
    png_path = os.path.join(config.FIGURES_DIR, "Figura13_qqplot_residuos.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura13_qqplot_residuos.pdf")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, format='pdf')
    plt.close()
    
    # Guardar en outputs
    excel_path = os.path.join(config.TABLES_DIR, "tabla_modelos_clasicos.xlsx")
    df_compare.to_excel(excel_path, index=False)
    
    logger.info(f"Reporte de comparación de modelos clásicos guardado en: {excel_path}")
    logger.info("ETAPA 7: Modelos Clásicos finalizada con éxito.")
    
    return poisson_res, nb_res, df_compare
