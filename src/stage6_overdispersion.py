import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage6")

def run_stage6(df_anual: pd.DataFrame) -> dict:
    """
    Etapa 6: Sobredispersión y Justificación Bayesiana
    - Calcula Media, Varianza y Ratio Varianza/Media de homicidios por subcircuito.
    - Ejecuta el test formal de sobredispersión de Cameron & Trivedi (1990).
    - Genera la justificación teórica automatizada basada en el p-valor.
    - Guarda los resultados en un archivo Excel.
    """
    logger.info("Iniciando ETAPA 6: Diagnóstico de Sobredispersión")
    
    # Agrupar homicidios totales por subcircuito
    df_sub = df_anual.groupby('cod_subcircu').agg({
        'homicidios': 'sum',
        'Area_km2': 'first'
    }).reset_index()
    
    y = df_sub['homicidios'].values
    
    mean_val = np.mean(y)
    var_val = np.var(y, ddof=1)
    ratio_val = var_val / mean_val
    
    logger.info(f"Métricas descriptivas del conteo:")
    logger.info(f" - Media: {mean_val:.4f}")
    logger.info(f" - Varianza: {var_val:.4f}")
    logger.info(f" - Ratio Varianza/Media (Dispersión): {ratio_val:.4f}")
    
    # Test formal de Cameron y Trivedi (1990)
    # Ajustar un modelo de regresión Poisson simple
    # Usamos Area_km2 como offset o simplemente ajustamos una constante
    df_sub['constante'] = 1.0
    poisson_model = smf.poisson('homicidios ~ 1', data=df_sub).fit(disp=0)
    mu_hat = poisson_model.predict(df_sub)
    
    # Auxiliar regression: z = alpha * w
    # z = ((y - mu_hat)^2 - y) / mu_hat
    # w = mu_hat (para la parametrización NB2 habitual)
    z = ((y - mu_hat)**2 - y) / mu_hat
    w = mu_hat
    
    # Regresión lineal sin intercepto
    aux_model = sm.OLS(z, w).fit()
    alpha_val = aux_model.params[0]
    t_stat = aux_model.tvalues[0]
    p_value = aux_model.pvalues[0] # Test de una cola sería p_value/2 si alpha > 0
    
    logger.info(f"Resultados de la prueba de Cameron & Trivedi:")
    logger.info(f" - Parámetro de sobredispersión (alpha): {alpha_val:.4f}")
    logger.info(f" - Estadístico t: {t_stat:.4f}")
    logger.info(f" - p-valor: {p_value:.6f}")
    
    # Justificación teórica automática para la tesis
    sobredispersado = ratio_val > 1.1 and p_value < 0.05
    if sobredispersado:
        justificacion = (
            f"Existe evidencia estadística robusta de SOBREDISPERSIÓN en la distribución espacial de los homicidios "
            f"(Varianza = {var_val:.2f} >> Media = {mean_val:.2f}, con un ratio de {ratio_val:.2f}). "
            f"El test formal de Cameron & Trivedi rechaza la hipótesis nula de equidispersión de Poisson de forma "
            f"significativa (alpha = {alpha_val:.4f}, t = {t_stat:.2f}, p-valor = {p_value:.6e}). "
            f"Esto justifica plenamente descartar el modelo frecuentista Poisson clásico y adoptar un modelo "
            f"con estructura Binomial Negativa que incorpore efectos aleatorios espaciales y temporales "
            f"bajo un enfoque Bayesiano Jerárquico."
        )
    else:
        justificacion = (
            f"Los datos no exhiben sobredispersión crítica significativa (Ratio = {ratio_val:.2f}, p-valor = {p_value:.4f}). "
            f"Sin embargo, debido a la estructura multinivel de los datos, la dependencia espacial espacial demostrada "
            f"en la Etapa 5 y la necesidad de modelar efectos temporales autoregresivos, el uso de un modelo jerárquico "
            f"bayesiano sigue siendo metodológicamente óptimo para regularizar las estimaciones en zonas pequeñas."
        )
        
    logger.info(f"Justificación Bayesiana: {justificacion}")
    
    # Estructurar resultados para guardar
    result_dict = {
        "Media": mean_val,
        "Varianza": var_val,
        "Ratio_Varianza_Media": ratio_val,
        "Alpha_Cameron_Trivedi": alpha_val,
        "Estadistico_t": t_stat,
        "p_valor": p_value,
        "Sobredispersión_Detectada": int(sobredispersado),
        "Justificación": justificacion
    }
    
    df_res = pd.DataFrame(list(result_dict.items()), columns=["Parámetro", "Valor"])
    
    df_ct = pd.DataFrame([
        {"Parámetro del Test": "Parámetro de Sobredispersión (alpha)", "Estimación": round(alpha_val, 4)},
        {"Parámetro del Test": "Estadístico t", "Estimación": round(t_stat, 4)},
        {"Parámetro del Test": "p-valor (Cameron & Trivedi 1990)", "Estimación": f"{p_value:.6e}" if p_value < 1e-4 else round(p_value, 6)},
        {"Parámetro del Test": "Decisión Metodológica", "Estimación": "Rechazo de H0 de Poisson (Sobredispersión Significativa)" if sobredispersado else "No rechaza H0"}
    ])
    
    # Guardar en outputs con dos pestañas claras
    excel_path = os.path.join(config.TABLES_DIR, "tabla_sobredispersion.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_res.to_excel(writer, sheet_name="Resumen", index=False)
        df_ct.to_excel(writer, sheet_name="Test Cameron-Trivedi", index=False)
    
    logger.info(f"Resultados de sobredispersión y test Cameron-Trivedi guardados en: {excel_path}")
    logger.info("ETAPA 6: Diagnóstico de Sobredispersión finalizado con éxito.")
    
    return result_dict
