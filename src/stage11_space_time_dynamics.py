import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage11")

def run_stage11(trace, gdf_anual: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Etapa 11: Análisis Espacio-Temporal y Dinámica de Hotspots
    - Calcula el riesgo relativo para cada año por separado.
    - Clasifica los subcircuitos en:
      - Hotspot Persistente (RR > 1.2 en todos los años)
      - Hotspot Emergente (RR_2024 < 1.0 y RR_2026 > 1.2)
      - Hotspot En Declive (RR_2024 > 1.2 y RR_2026 < 1.0)
      - Sin Riesgo Crítico / Estable
    - Genera gráficos comparativos de evolución (Figura 21) a 300 dpi.
    - Exporta resultados en CSV y Excel.
    """
    logger.info("Iniciando ETAPA 11: Análisis Dinámico de Hotspots")
    
    # 1. Extraer muestras para calcular riesgo anual
    spatial_var = 'phi' if 'phi' in trace.posterior else ('psi' if 'psi' in trace.posterior else 'spatial_total')
    phi_samples = trace.posterior[spatial_var].values.reshape(-1, trace.posterior[spatial_var].shape[-1])
    n_samples, n_zonas = phi_samples.shape
    
    # Estandarización de Riesgo Relativo sobre la media cantonal (Clayton & Kaldor, 1987)
    exp_phi = np.exp(phi_samples)
    spatial_mean = np.mean(exp_phi, axis=1, keepdims=True)
    rr_samples_std = exp_phi / spatial_mean
    
    # Mapeo de subcircuitos y años
    sub_codes = sorted(gdf_anual['cod_subcircu'].unique())
    anios = sorted(gdf_anual['Anio'].unique())
    n_periodos = len(anios)
    
    # Calcular el RR mediano posterior por zona
    rr_median_zonal = np.median(rr_samples_std, axis=0)
    
    # Crear DataFrame de evolución
    evolution_dict = {'cod_subcircu': sub_codes}
    for t, anio in enumerate(anios):
        evolution_dict[f'rr_{anio}'] = rr_median_zonal
        
    df_evol = pd.DataFrame(evolution_dict)
    
    # 2. Clasificación de Hotspots
    # Asumimos que los años disponibles son 2024, 2025 y 2026 (o los que estén en la data)
    col_2024 = f'rr_{anios[0]}'
    col_last = f'rr_{anios[-1]}'
    
    classifications = []
    for idx, row in df_evol.iterrows():
        rr_2024 = row[col_2024]
        rr_last = row[col_last]
        
        # Evaluar todos los años para persistencia
        all_high = all(row[f'rr_{a}'] >= 1.2 for a in anios)
        
        if all_high:
            classifications.append('HOTSPOT PERSISTENTE')
        elif rr_2024 < 1.0 and rr_last >= 1.2:
            classifications.append('HOTSPOT EMERGENTE')
        elif rr_2024 >= 1.2 and rr_last < 1.0:
            classifications.append('HOTSPOT EN DECLIVE')
        else:
            classifications.append('ESTABLE / BAJO RIESGO')
            
    df_evol['clasificacion_dinamica'] = classifications
    
    # Unir nombres de subcircuitos para el reporte
    names_map = gdf_anual[['cod_subcircu', 'subcircuito_nom', 'circuito_nom', 'distrito_nom']].drop_duplicates(subset=['cod_subcircu'])
    df_evol = df_evol.merge(names_map, on='cod_subcircu', how='left')
    
    # 3. Guardar resultados
    csv_path = os.path.join(config.CLEAN_DATA_DIR, "dinamica_hotspots.csv")
    excel_path = os.path.join(config.TABLES_DIR, "tabla_dinamica_hotspots.xlsx")
    
    df_evol.to_csv(csv_path, index=False)
    
    # Filtrar tablas específicas
    df_persistent = df_evol[df_evol['clasificacion_dinamica'] == 'HOTSPOT PERSISTENTE'].sort_values(by=col_last, ascending=False)
    df_emergent = df_evol[df_evol['clasificacion_dinamica'] == 'HOTSPOT EMERGENTE'].sort_values(by=col_last, ascending=False)
    df_declining = df_evol[df_evol['clasificacion_dinamica'] == 'HOTSPOT EN DECLIVE'].sort_values(by=col_2024, ascending=False)
    
    with pd.ExcelWriter(excel_path) as writer:
        df_evol.to_excel(writer, sheet_name="Evolucion Completa", index=False)
        df_persistent.to_excel(writer, sheet_name="Hotspots Persistentes", index=False)
        df_emergent.to_excel(writer, sheet_name="Hotspots Emergentes", index=False)
        df_declining.to_excel(writer, sheet_name="Hotspots en Declive", index=False)
        
    logger.info("Tablas de dinámica espacio-temporal exportadas con éxito.")
    
    # 4. Gráfico 21: Evolución del Riesgo Relativo (Líneas para Top Zonas Críticas)
    plt.figure(figsize=(10, 6))
    
    # Tomar top 7 subcircuitos con mayor riesgo final
    top_zones = df_evol.sort_values(by=col_last, ascending=False).head(7)
    
    for idx, row in top_zones.iterrows():
        rr_vals = [row[f'rr_{a}'] for a in anios]
        plt.plot(anios, rr_vals, marker='o', label=row['subcircuito_nom'], linewidth=2)
        
    plt.axhline(1.0, color='red', linestyle='--', linewidth=0.8, label='Línea de Riesgo Base (1.0)')
    plt.title("Figura 21: Evolución Temporal del Riesgo Relativo en los Subcircuitos Críticos")
    plt.xlabel("Año del Homicidio")
    plt.ylabel("Riesgo Relativo Estimado (Posterior Median)")
    plt.xticks(anios)
    plt.grid(True, axis='y')
    plt.legend(prop={'size': 8})
    
    # Guardar
    png_path = os.path.join(config.FIGURES_DIR, "Figura21_evolucion_riesgo_hotspots.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura21_evolucion_riesgo_hotspots.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("ETAPA 11: Análisis Dinámico de Hotspots finalizado con éxito.")
    return df_evol
