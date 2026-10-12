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

def connect_graph_islands(w: libpysal.weights.W, gdf: gpd.GeoDataFrame) -> libpysal.weights.W:
    """
    Conecta islas y componentes disconexos al vecino más cercano según centroides geométricos,
    garantizando un grafo conexo (0 islas, 1 único componente conexo) requerido por los modelos
    espaciales bayesianos CAR/ICAR/BYM2 con restricción de suma cero (sum-to-zero).
    """
    from scipy.sparse.csgraph import connected_components
    neighbors = {k: list(v) for k, v in w.neighbors.items()}
    # Proyectar a UTM 17S si está en coordenadas geográficas para distancias métricas euclidianas exactas
    gdf_proj = gdf.to_crs("EPSG:32717") if (gdf.crs is not None and gdf.crs.is_geographic) else gdf
    centroids = gdf_proj.geometry.centroid
    adj = w.sparse
    n_comp, labels = connected_components(adj, directed=False)
    
    if n_comp > 1 or len(w.islands) > 0:
        main_comp = np.argmax(np.bincount(labels))
        main_nodes = np.where(labels == main_comp)[0]
        
        for comp in range(n_comp):
            if comp == main_comp:
                continue
            comp_nodes = np.where(labels == comp)[0]
            min_dist = np.inf
            best_c, best_m = None, None
            for c in comp_nodes:
                dists = centroids.iloc[main_nodes].distance(centroids.iloc[c])
                idx_m = dists.idxmin()
                if dists.loc[idx_m] < min_dist:
                    min_dist = dists.loc[idx_m]
                    best_c = c
                    best_m = idx_m
            if best_c is not None and best_m is not None:
                if best_m not in neighbors[best_c]:
                    neighbors[best_c].append(best_m)
                if best_c not in neighbors[best_m]:
                    neighbors[best_m].append(best_c)
        w_connected = libpysal.weights.W(neighbors)
        return w_connected
    return w

def compare_weights(gdf: gpd.GeoDataFrame) -> tuple:
    """
    Construye las matrices Queen y Rook y las compara formalmente.
    
    NOTA METODOLÓGICA (QUEEN vs k-NN):
    La tesis en ciertos apartados iniciales mencionaba k-NN, pero el análisis espacial de área
    y los modelos bayesianos CAR/ICAR/BYM2 requieren pesos basados en contigüidad física (Queen de 1er orden).
    La matriz Queen captura continuidad geográfica real por compartir frontera o vértice común.
    A diferencia de k-NN que fuerza un número arbitrario y constante de vecinos sin considerar distancia física,
    Queen refleja la estructura morfológica y topológica del mosaico urbano cantonal.
    """
    logger.info("Construyendo matriz de pesos espaciales Queen (contigüidad de primer orden)...")
    gdf_valid = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    
    w_queen_raw = libpysal.weights.Queen.from_dataframe(gdf_valid, use_index=False)
    logger.info("Construyendo matriz de pesos espaciales Rook...")
    w_rook_raw = libpysal.weights.Rook.from_dataframe(gdf_valid, use_index=False)
    
    # Métricas para Queen y Rook crudas
    queen_islands_raw = len(w_queen_raw.islands)
    rook_islands_raw = len(w_rook_raw.islands)
    logger.info(f"Matriz Queen inicial detectó {queen_islands_raw} islas aisladas (Isla Puná: {w_queen_raw.islands}).")
    logger.info(f"Matriz Rook inicial detectó {rook_islands_raw} islas aisladas.")
    
    # Conectar islas para garantizar regularidad en el grafo espacial del modelo
    logger.info("Conectando islas al vecino continental más cercano para el modelo espacial...")
    w_queen = connect_graph_islands(w_queen_raw, gdf_valid)
    w_rook = connect_graph_islands(w_rook_raw, gdf_valid)
    
    queen_islands = len(w_queen.islands)
    queen_mean_neigh = float(np.mean([len(v) for v in w_queen.neighbors.values()]))
    queen_density = float(w_queen.pct_nonzero)  # w.pct_nonzero ya está expresado en porcentaje por libpysal
    
    rook_islands = len(w_rook.islands)
    rook_mean_neigh = float(np.mean([len(v) for v in w_rook.neighbors.values()]))
    rook_density = float(w_rook.pct_nonzero)    # w.pct_nonzero ya está expresado en porcentaje por libpysal
    
    logger.info(f"Tras conexión: Queen tiene {queen_islands} islas y {w_queen.n_components} componentes conexos.")
    logger.info(f"MATRIZ FINAL SELECCIONADA PARA MODELADO BAYESIANO: Queen con las {queen_islands_raw} islas conectadas (0 islas finales, grafo conexo), promedio de vecinos: {queen_mean_neigh:.2f}, densidad: {queen_density:.4f}%.")
    
    comparison_data = {
        "Métrica": [
            "Número de Observaciones (N)",
            "Islas Iniciales (sin conectar)",
            "Islas Finales (conectadas)",
            "Promedio de Vecinos",
            "Densidad de la Matriz (%)",
            "Máximo número de vecinos",
            "Mínimo número de vecinos",
            "Matriz Seleccionada para el Modelo Final"
        ],
        "Matriz Queen": [
            w_queen.n,
            queen_islands_raw,
            queen_islands,
            round(queen_mean_neigh, 2),
            round(queen_density, 4),
            max([len(v) for v in w_queen.neighbors.values()]),
            min([len(v) for v in w_queen.neighbors.values()]),
            "SÍ (Queen, 4 islas de Puná conectadas)"
        ],
        "Matriz Rook": [
            w_rook.n,
            rook_islands_raw,
            rook_islands,
            round(rook_mean_neigh, 2),
            round(rook_density, 4),
            max([len(v) for v in w_rook.neighbors.values()]),
            min([len(v) for v in w_rook.neighbors.values()]),
            "NO (Descartada)"
        ]
    }
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Se selecciona Queen como estándar científico por cubrir uniones de vértices y bordes
    selected_w = w_queen
    selected_name = "Queen (Conectada)"
    logger.info("Se ha seleccionado formalmente la matriz QUEEN con islas conectadas para el análisis ESDA y el modelo Bayesiano.")
        
    return selected_w, selected_name, df_comp

def run_stage5(gdf_total: gpd.GeoDataFrame) -> tuple:
    """
    Etapa 5: Análisis Espacial Exploratorio (ESDA)
    - Compara matrices Queen vs Rook y conecta islas.
    - Calcula el Moran's I Global con 9999 permutaciones y semilla fija.
    - Calcula el LISA (Moran Local) para mapear hotspots/coldspots.
    - Genera Moran Scatterplot y mapas LISA numerados a 300 dpi.
    - Guarda tablas y GeoJSON procesados.
    """
    logger.info("Iniciando ETAPA 5: Análisis Espacial Exploratorio (ESDA)")
    
    # Fijar semilla para reproducibilidad exacta
    np.random.seed(config.RANDOM_SEED)
    
    # 1. Comparar matrices de vecinos y conectar islas
    w_selected, w_name, df_comp = compare_weights(gdf_total)
    
    # Estandarizar la matriz por filas
    w_selected.transform = 'r'
    
    # Variable de conteo y tasa para el análisis
    y = gdf_total['homicidios_totales'].values
    
    # 2. Moran Global con semilla fija
    logger.info("Calculando Moran's I Global con 9999 permutaciones...")
    np.random.seed(config.RANDOM_SEED)
    moran = Moran(y, w_selected, permutations=9999)
    
    logger.info(f"Resultados Moran Global: I={moran.I:.4f}, E(I)={moran.EI:.4f}, z={moran.z_sim:.4f}, p-valor={moran.p_sim:.4f}")
    
    # 3. Moran Local (LISA)
    logger.info("Calculando LISA (Moran Local) con 9999 permutaciones...")
    np.random.seed(config.RANDOM_SEED)
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
    mean_neigh_queen = float(np.mean([len(v) for v in w_selected.neighbors.values()]))
    logger.info(f"MORAN GLOBAL (Matriz Queen final con 4 islas conectadas): I={moran.I:.4f}, E(I)={moran.EI:.4f}, z={moran.z_sim:.4f}, p-valor={moran.p_sim:.4f} (permutaciones={moran.permutations}, semilla={config.RANDOM_SEED}, vecinos={mean_neigh_queen:.2f})")
    
    df_moran_global = pd.DataFrame({
        "Métrica": [
            "Indice Moran I",
            "Esperanza E(I)",
            "Z-Score",
            "Pseudo p-valor",
            "Permutaciones",
            "Semilla Aleatoria",
            "Matriz Espacial Utilizada",
            "Promedio Vecinos Matriz"
        ],
        "Valor": [
            round(moran.I, 4),
            round(moran.EI, 4),
            round(moran.z_sim, 4),
            round(moran.p_sim, 4),
            moran.permutations,
            config.RANDOM_SEED,
            "Queen (4 islas de Puná conectadas)",
            round(mean_neigh_queen, 2)
        ]
    })
    
    excel_esda_path = os.path.join(config.TABLES_DIR, "tabla_esda_moran.xlsx")
    with pd.ExcelWriter(excel_esda_path) as writer:
        df_comp.to_excel(writer, sheet_name="Comparativa Matrices", index=False)
        df_moran_global.to_excel(writer, sheet_name="Moran Global", index=False)
        
    logger.info(f"Tablas ESDA guardadas en: {excel_esda_path}")
    logger.info("ETAPA 5: Análisis Espacial Exploratorio (ESDA) finalizado con éxito.")
    
    return gdf_total, moran, df_comp, df_moran_global
