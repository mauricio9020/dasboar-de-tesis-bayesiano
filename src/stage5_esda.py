import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import libpysal
from esda.moran import Moran, Moran_Local
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage5")

def compare_weights(gdf: gpd.GeoDataFrame) -> tuple:
    """
    Construye las matrices Queen y Rook y las compara formalmente.
    Retorna la matriz seleccionada y un resumen tabular.
    """
    logger.info("Construyendo matriz de pesos espaciales Queen...")
    # Asegurar que no haya geometrías nulas o vacías
    gdf_valid = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    
    w_queen = libpysal.weights.Queen.from_dataframe(gdf_valid, use_index=False)
    logger.info("Construyendo matriz de pesos espaciales Rook...")
    w_rook = libpysal.weights.Rook.from_dataframe(gdf_valid, use_index=False)
    
    # Métricas para Queen
    queen_islands = len(w_queen.islands)
    queen_mean_neigh = np.mean([len(v) for v in w_queen.neighbors.values()])
    queen_density = w_queen.pct_nonzero
    
    # Métricas para Rook
    rook_islands = len(w_rook.islands)
    rook_mean_neigh = np.mean([len(v) for v in w_rook.neighbors.values()])
    rook_density = w_rook.pct_nonzero
    
    comparison_data = {
        "Métrica": [
            "Número de Observaciones (N)",
            "Número de Islas (Aislados)",
            "Promedio de Vecinos",
            "Densidad de la Matriz (%)",
            "Máximo número de vecinos",
            "Mínimo número de vecinos"
        ],
        "Matriz Queen": [
            w_queen.n,
            queen_islands,
            round(queen_mean_neigh, 2),
            round(queen_density * 100, 4),
            max([len(v) for v in w_queen.neighbors.values()]),
            min([len(v) for v in w_queen.neighbors.values()])
        ],
        "Matriz Rook": [
            w_rook.n,
            rook_islands,
            round(rook_mean_neigh, 2),
            round(rook_density * 100, 4),
            max([len(v) for v in w_rook.neighbors.values()]),
            min([len(v) for v in w_rook.neighbors.values()])
        ]
    }
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Selección automática: Se prefiere Queen porque permite uniones de esquinas
    # y suele generar menos islas en tramas urbanas irregulares.
    if queen_islands <= rook_islands:
        selected_w = w_queen
        selected_name = "Queen"
        logger.info("Se ha seleccionado la matriz QUEEN por tener menor o igual número de islas.")
    else:
        selected_w = w_rook
        selected_name = "Rook"
        logger.info("Se ha seleccionado la matriz ROOK por menor número de islas.")
        
    return selected_w, selected_name, df_comp

def run_stage5(gdf_total: gpd.GeoDataFrame) -> tuple:
    """
    Etapa 5: Análisis Espacial Exploratorio (ESDA)
    - Compara matrices Queen vs Rook y selecciona la mejor.
    - Calcula el Moran's I Global con 9999 permutaciones.
    - Calcula el LISA (Moran Local) para mapear hotspots/coldspots.
    - Genera Moran Scatterplot y mapas LISA numerados a 300 dpi.
    - Guarda tablas y GeoJSON procesados.
    """
    logger.info("Iniciando ETAPA 5: Análisis Espacial Exploratorio (ESDA)")
    
    # 1. Comparar matrices de vecinos
    w_selected, w_name, df_comp = compare_weights(gdf_total)
    
    # Estandarizar la matriz por filas
    w_selected.transform = 'r'
    
    # Variable de conteo y tasa para el análisis
    y = gdf_total['homicidios_totales'].values
    
    # 2. Moran Global
    logger.info("Calculando Moran's I Global con 9999 permutaciones...")
    moran = Moran(y, w_selected, permutations=9999)
    
    logger.info(f"Resultados Moran Global: I={moran.I:.4f}, E(I)={moran.EI:.4f}, z={moran.z_sim:.4f}, p-valor={moran.p_sim:.4f}")
    
    # 3. Moran Local (LISA)
    logger.info("Calculando LISA (Moran Local) con 9999 permutaciones...")
    lisa = Moran_Local(y, w_selected, permutations=9999)
    
    # Clasificación LISA
    # lisa.q contiene el cuadrante: 1=HH, 2=LH, 3=LL, 4=HL
    # p_sim contiene el p-valor
    quad_labels = {1: 'HIGH-HIGH (HOTSPOT)', 2: 'LOW-HIGH (OUTLIER)', 3: 'LOW-LOW (COLDSPOT)', 4: 'HIGH-LOW (OUTLIER)'}
    
    lisa_cats = []
    for i in range(len(gdf_total)):
        p_val = lisa.p_sim[i]
        q_val = lisa.q[i]
        if p_val < 0.05:
            lisa_cats.append(quad_labels[q_val])
        else:
            lisa_cats.append('NOT SIGNIFICANT')
            
    gdf_total['lisa_category'] = lisa_cats
    gdf_total['lisa_I'] = lisa.Is
    gdf_total['lisa_p_value'] = lisa.p_sim
    
    # Guardar GeoJSON final con la clasificación espacial
    esda_geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_esda.geojson")
    gdf_total.to_crs("EPSG:4326").to_file(esda_geojson_path, driver='GeoJSON')
    
    # 4. Gráfico 11: Moran Scatterplot
    plt.figure(figsize=(8, 6))
    y_std = (y - np.mean(y)) / np.std(y)
    y_lag = libpysal.weights.spatial_lag.lag_spatial(w_selected, y)
    y_lag_std = (y_lag - np.mean(y_lag)) / np.std(y_lag)
    
    sns.regplot(x=y_std, y=y_lag_std, ci=None, line_kws={'color': 'darkred', 'linewidth': 1.5},
                scatter_kws={'color': 'navy', 'alpha': 0.6, 'edgecolor': 'white'})
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title(f"Figura 11: Moran Scatterplot de Homicidios (Moran's I = {moran.I:.4f})")
    plt.xlabel("Homicidios Estandarizados ($z_i$)")
    plt.ylabel("Lag Espacial Estandarizado ($W z_i$)")
    
    # Guardar
    png_path = os.path.join(config.FIGURES_DIR, "Figura11_moran_scatterplot.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura11_moran_scatterplot.pdf")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, format='pdf')
    plt.close()
    
    # 5. Gráfico 12: Mapa LISA
    plt.figure(figsize=(10, 10))
    # Definir colores fijos para cada categoría
    colors_dict = {
        'HIGH-HIGH (HOTSPOT)': '#d7191c',  # Rojo
        'LOW-LOW (COLDSPOT)': '#2c7bb6',   # Azul
        'HIGH-LOW (OUTLIER)': '#fdae61',   # Naranja claro
        'LOW-HIGH (OUTLIER)': '#abd9e9',   # Azul claro
        'NOT SIGNIFICANT': '#f7f7f7'       # Gris claro
    }
    
    # Dibujar mapa base
    gdf_total.plot(color='#eeeeee', edgecolor='#cccccc', linewidth=0.5, ax=plt.gca())
    
    # Graficar por categoría
    for cat, color in colors_dict.items():
        sub_gdf = gdf_total[gdf_total['lisa_category'] == cat]
        if not sub_gdf.empty:
            sub_gdf.plot(color=color, edgecolor='black', linewidth=0.4, ax=plt.gca(), label=cat)
            
    plt.title("Figura 12: Mapa de Agrupamiento LISA (Hotspots y Coldspots)")
    
    # Leyenda personalizada
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=color, label=cat) for cat, color in colors_dict.items()]
    plt.legend(handles=patches, loc='lower left', prop={'size': 8})
    plt.axis('off')
    
    # Guardar
    png_map_path = os.path.join(config.FIGURES_DIR, "Figura12_mapa_lisa.png")
    pdf_map_path = os.path.join(config.FIGURES_DIR, "Figura12_mapa_lisa.pdf")
    plt.savefig(png_map_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_map_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    # Guardar métricas del Moran Global a Excel
    df_moran_global = pd.DataFrame({
        "Métrica": ["Indice Moran I", "Esperanza E(I)", "Z-Score", "Pseudo p-valor (9999 perm)"],
        "Valor": [moran.I, moran.EI, moran.z_sim, moran.p_sim]
    })
    
    excel_esda_path = os.path.join(config.TABLES_DIR, "tabla_esda_moran.xlsx")
    with pd.ExcelWriter(excel_esda_path) as writer:
        df_comp.to_excel(writer, sheet_name="Comparativa Matrices", index=False)
        df_moran_global.to_excel(writer, sheet_name="Moran Global", index=False)
        
    logger.info(f"Tablas ESDA guardadas en: {excel_esda_path}")
    logger.info("ETAPA 5: Análisis Espacial Exploratorio (ESDA) finalizado con éxito.")
    
    return gdf_total, moran, df_comp, df_moran_global
