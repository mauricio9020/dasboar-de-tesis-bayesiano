import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage10")

def run_stage10(trace, gdf_total: gpd.GeoDataFrame, gdf_anual: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Etapa 10: Estimación de Riesgo Relativo (RR)
    - Extrae muestras posteriores de los coeficientes de efectos espaciales.
    - Calcula el RR mediano posterior y sus intervalos de credibilidad (HDI 95%).
    - Calcula la probabilidad posterior de excedencia P(RR > 1.5).
    - Genera mapas estáticos suavizados de riesgo (Figura 20) y los exporta a 300 dpi.
    - Guarda los resultados tabulares en un CSV y Excel.
    """
    logger.info("Iniciando ETAPA 10: Estimación de Riesgo Relativo y Mapas")
    
    # 1. Extraer muestras de la posterior del efecto espacial
    # Identificar la variable espacial del modelo (phi, psi en BYM2, o spatial_total)
    spatial_var = 'phi' if 'phi' in trace.posterior else ('psi' if 'psi' in trace.posterior else 'spatial_total')
    logger.info(f"Extrayendo efecto espacial posterior desde la variable: '{spatial_var}'")
    
    phi_samples = trace.posterior[spatial_var].values.reshape(-1, trace.posterior[spatial_var].shape[-1])  # shape (n_samples, n_zonas)
    n_samples, n_zonas = phi_samples.shape
    
    # 2. DEFINICIÓN RIGUROSA DE RIESGO RELATIVO (RR):
    # En epidemiología espacial (Clayton & Kaldor, 1987; Besag et al., 1991), el Riesgo Relativo RR_i
    # compara el riesgo del subcircuito i frente a la referencia basal promedio cantonal (RR = 1.0).
    # Se calcula muestra a muestra estandarizando por la media espacial de los efectos exponenciales:
    # RR_i^(s) = exp(phi_i^(s)) / [ (1/N) * sum_k exp(phi_k^(s)) ]
    # De este modo, un subcircuito con RR = 1.0 representa riesgo idéntico a la media cantonal,
    # RR = 1.5 representa un 50% más de riesgo que el promedio cantonal, y P(RR > 1.5) mide la
    # probabilidad posterior de riesgo crítico sobre la referencia cantonal estandarizada.
    
    exp_phi = np.exp(phi_samples)  # (n_samples, n_zonas)
    spatial_mean = np.mean(exp_phi, axis=1, keepdims=True)  # (n_samples, 1)
    rr_samples = exp_phi / spatial_mean  # (n_samples, n_zonas)
    
    logger.info(f"Riesgo Relativo calculado para {n_zonas} subcircuitos sobre {n_samples} muestras posteriores.")
    logger.info(f"Media global de RR_samples verificada: {np.mean(rr_samples):.4f} (referencia cantonal = 1.0000).")
        
    # Extraer estimaciones posteriores puntuales y de incertidumbre
    rr_median = np.median(rr_samples, axis=0)
    rr_mean = np.mean(rr_samples, axis=0)
    rr_lower = np.percentile(rr_samples, 2.5, axis=0)
    rr_upper = np.percentile(rr_samples, 97.5, axis=0)
    
    # Probabilidad posterior de excedencia de riesgo crítico P(RR > 1.5)
    exceedance_threshold = 1.5
    prob_exceedance = np.mean(rr_samples > exceedance_threshold, axis=0)
    
    # 3. Mapear resultados a los subcircuitos
    # Obtenemos los códigos ordenados
    sub_codes = sorted(gdf_total['cod_subcircu'].unique())
    
    df_rr = pd.DataFrame({
        'cod_subcircu': sub_codes,
        'rr_estimado_mediana': rr_median,
        'rr_estimado_media': rr_mean,
        'rr_limite_inf_95': rr_lower,
        'rr_limite_sup_95': rr_upper,
        'probabilidad_excedencia': prob_exceedance
    })
    
    # 4. Clasificación de Riesgo Relativo por Quintiles
    q = df_rr['rr_estimado_mediana'].quantile([0.2, 0.4, 0.6, 0.8]).values
    labels = [
        f'Muy Bajo (RR ≤ {q[0]:.2f})',
        f'Bajo ({q[0]:.2f} - {q[1]:.2f})',
        f'Medio ({q[1]:.2f} - {q[2]:.2f})',
        f'Alto ({q[2]:.2f} - {q[3]:.2f})',
        f'Muy Alto (RR > {q[3]:.2f})'
    ]
    df_rr['rr_categoria'] = pd.qcut(df_rr['rr_estimado_mediana'], q=5, labels=labels)
    
    # Unir con gdf_total
    gdf_rr = gdf_total.merge(df_rr, on='cod_subcircu', how='left')
    
    # Guardar resultados
    csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
    excel_path = os.path.join(config.TABLES_DIR, "tabla_riesgo_relativo.xlsx")
    
    # Unir metadatos de nombres a df_rr y ordenar por RR descendente
    df_names = gdf_total[['cod_subcircu', 'subcircuito_nom', 'circuito_nom', 'distrito_nom', 'homicidios_totales']].drop_duplicates(subset=['cod_subcircu'])
    df_rr_names = df_names.merge(df_rr, on='cod_subcircu', how='right').sort_values(by='rr_estimado_mediana', ascending=False)
    
    df_rr_names.to_csv(csv_path, index=False)
    # Top 20 subcircuitos más calientes para la tesis
    df_top_rr = gdf_rr[['cod_subcircu', 'subcircuito_nom', 'circuito_nom', 'distrito_nom', 'homicidios_totales', 'rr_estimado_mediana', 'probabilidad_excedencia']].sort_values(by='rr_estimado_mediana', ascending=False)
    
    with pd.ExcelWriter(excel_path) as writer:
        df_rr_names.to_excel(writer, sheet_name="Riesgos Relativos", index=False)
        df_top_rr.head(20).to_excel(writer, sheet_name="Top 20 Criticos", index=False)
        
    logger.info(f"Tablas de riesgo guardadas en CSV y Excel con nombres descriptivos de subcircuitos.")
    
    # Guardar GeoJSON final con todos los datos integrados para Leaflet
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
    gdf_rr.to_crs("EPSG:4326").to_file(geojson_path, driver='GeoJSON')
    logger.info(f"GeoJSON completo guardado en: {geojson_path}")
    
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches
    
    colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    custom_cmap = ListedColormap(colors)
    
    # Cargar límites parroquiales disueltos para superposición de referencia
    gdf_parr = None
    if os.path.exists(config.SHAPEFILE_PATH):
        try:
            gdf_shp = gpd.read_file(config.SHAPEFILE_PATH)
            gdf_gye_shp = gdf_shp[gdf_shp['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
            if gdf_gye_shp.crs is None:
                gdf_gye_shp = gdf_gye_shp.set_crs("EPSG:32717")
            gdf_parr = gdf_gye_shp.dissolve(by='DPA_DESPAR', as_index=False)
        except Exception as e:
            logger.warning(f"No se pudo disolver el shapefile por parroquias para la visualización: {e}")

    # 5. Generar Figura 20: Mapas de Riesgo Relativo y Excedencia (2 Paneles)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Mapa A: Riesgo Relativo Median Posterior Clasificado
    gdf_rr.plot(column='rr_categoria', cmap=custom_cmap, legend=True,
                legend_kwds={
                    'title': 'Categorías de Riesgo Relativo (RR)',
                    'loc': 'lower left',
                    'fontsize': 9,
                    'title_fontsize': 10,
                    'frameon': True,
                    'facecolor': 'white',
                    'edgecolor': '#cccccc'
                }, ax=ax1,
                edgecolor='#666666', linewidth=0.2)
    ax1.set_title("A: Estimación Suavizada de Riesgo Relativo (Clasificado)", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Superponer límites parroquiales si están disponibles
    if gdf_parr is not None:
        gdf_parr_proj = gdf_parr.to_crs(gdf_rr.crs)
        gdf_parr_proj.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=0.8, linestyle='--', zorder=3)
        
    # Mapa B: Probabilidades de Excedencia P(RR > 1.5)
    gdf_rr.plot(column='probabilidad_excedencia', cmap='YlOrRd', legend=True,
                legend_kwds={'label': 'P(RR > 1.5)'}, ax=ax2,
                edgecolor='#666666', linewidth=0.2)
    ax2.set_title("B: Probabilidad Posterior de Excedencia $P(RR > 1.5 \,|\, \\text{Datos})$", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.suptitle("Figura 20: Cartografía Bayesiana Suavizada de Riesgo de Homicidios", fontsize=14, fontweight='bold', y=0.98)
    
    # Guardar Figura 20
    png_path = os.path.join(config.FIGURES_DIR, "Figura20_mapas_riesgo_bayesiano.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura20_mapas_riesgo_bayesiano.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    # 6. Generar el Mapa Standalone Impecable (1 Panel, Proyección UTM con Escala, Norte, Parroquias y Leyenda)
    try:
        # Proyectar a UTM para medidas métricas exactas
        gdf_rr_utm = gdf_rr.to_crs("EPSG:32717")
        
        fig_st, ax_st = plt.subplots(figsize=(12, 14))
        
        # Graficar subcircuitos
        gdf_rr_utm.plot(
            column='rr_categoria',
            cmap=custom_cmap,
            legend=True,
            legend_kwds={
                'title': 'Riesgo Relativo Posterior (RR)',
                'loc': 'lower left',
                'fontsize': 11,
                'title_fontsize': 12,
                'frameon': True,
                'facecolor': 'white',
                'edgecolor': '#cccccc'
            },
            ax=ax_st,
            edgecolor='#666666',
            linewidth=0.2,
            alpha=0.95
        )
        
        # Graficar límites parroquiales disueltos
        if gdf_parr is not None:
            gdf_parr_utm = gdf_parr.to_crs("EPSG:32717")
            gdf_parr_utm.plot(
                ax=ax_st,
                facecolor='none',
                edgecolor='#111111',
                linewidth=1.3,
                linestyle='--',
                zorder=3
            )
            
            # Anotaciones de parroquias en sus centroides
            for idx, row in gdf_parr_utm.iterrows():
                name = row['DPA_DESPAR'].split('(')[0].strip()
                if "GUAYAQUIL" in name:
                    name = "GUAYAQUIL (URBANO)"
                centroid = row.geometry.centroid
                
                cx, cy = centroid.x, centroid.y
                if "PUNA" in name:
                    cy += 5000
                elif "MORRO" in name:
                    cx -= 2000
                elif "POSORJA" in name:
                    cy -= 2000
                    
                ax_st.text(
                    cx, cy, name,
                    fontsize=9,
                    fontweight='bold',
                    color='black',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", lw=0.5, alpha=0.85),
                    zorder=4
                )
        
        # Ajustar límites de visualización de acuerdo a la geometría
        xmin, ymin, xmax, ymax = gdf_rr_utm.total_bounds
        x_pad = (xmax - xmin) * 0.02
        y_pad = (ymax - ymin) * 0.02
        ax_st.set_xlim(xmin - x_pad, xmax + x_pad)
        ax_st.set_ylim(ymin - y_pad, ymax + y_pad)
        
        # Dibujar escala (10 km) en la esquina inferior derecha
        scale_len = 10000
        scale_x_start = xmax - scale_len - 10000
        scale_y = ymin + 15000
        
        tick_height = (ymax - ymin) * 0.008
        text_offset = tick_height * 1.5
        
        scale_box = patches.Rectangle(
            (scale_x_start - 2000, scale_y - tick_height * 2),
            scale_len + 4000,
            tick_height * 5,
            linewidth=0.5,
            edgecolor='#cccccc',
            facecolor='white',
            alpha=0.85,
            zorder=4
        )
        ax_st.add_patch(scale_box)
        
        ax_st.plot([scale_x_start, scale_x_start + scale_len], [scale_y, scale_y], color='black', linewidth=2.5, zorder=5)
        ax_st.plot([scale_x_start, scale_x_start], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
        ax_st.plot([scale_x_start + scale_len/2, scale_x_start + scale_len/2], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
        ax_st.plot([scale_x_start + scale_len, scale_x_start + scale_len], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
        
        ax_st.text(scale_x_start, scale_y + text_offset, '0', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
        ax_st.text(scale_x_start + scale_len/2, scale_y + text_offset, '5 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
        ax_st.text(scale_x_start + scale_len, scale_y + text_offset, '10 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
        
        # Dibujar flecha del norte en la esquina superior izquierda
        arrow_x = xmin + 15000
        arrow_y = ymax - 20000
        
        rect_n = patches.Rectangle(
            (arrow_x - 5000, arrow_y - 7000),
            10000,
            14000,
            linewidth=0.5,
            edgecolor='#cccccc',
            facecolor='white',
            alpha=0.85,
            zorder=4
        )
        ax_st.add_patch(rect_n)
        
        ax_st.annotate(
            'N', xy=(arrow_x, arrow_y + 4000), xytext=(arrow_x, arrow_y - 4000),
            arrowprops=dict(facecolor='black', width=3, headwidth=10, shrink=0.05),
            ha='center', va='center', fontsize=14, fontweight='bold', color='black',
            zorder=5
        )
        
        # Recuadro estético
        rect_border = patches.Rectangle(
            (xmin - x_pad/2, ymin - y_pad/2),
            (xmax - xmin) + x_pad,
            (ymax - ymin) + y_pad,
            linewidth=1,
            edgecolor='#aaaaaa',
            facecolor='none',
            zorder=10
        )
        ax_st.add_patch(rect_border)
        
        ax_st.set_title("MAPA DE RIESGO RELATIVO BAYESIANO (CLASIFICADO)\nCanton Guayaquil: Parroquias y Subcircuitos", fontsize=15, fontweight='bold', pad=20)
        ax_st.axis('off')
        
        mapa_png_path = os.path.join(config.FIGURES_DIR, "mapa_riesgo_relativo_clasificado.png")
        mapa_pdf_path = os.path.join(config.FIGURES_DIR, "mapa_riesgo_relativo_clasificado.pdf")
        plt.savefig(mapa_png_path, dpi=300, bbox_inches='tight')
        plt.savefig(mapa_pdf_path, format='pdf', bbox_inches='tight')
        plt.close()
        logger.info(f"Mapa standalone clasificado guardado en: {mapa_png_path}")
    except Exception as e:
        logger.error(f"Error al generar el mapa standalone: {e}", exc_info=True)
        
    logger.info("ETAPA 10: Estimación de Riesgo Relativo y Mapas finalizada con éxito.")
    return gdf_rr
