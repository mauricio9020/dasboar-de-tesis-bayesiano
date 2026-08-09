import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def main():
    print("Cargando datos...")
    # Load subcircuits GeoJSON
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
    gdf_total = gpd.read_file(geojson_path)
    
    # Load original shapefile to dissolve by parish
    gdf_shp = gpd.read_file(config.SHAPEFILE_PATH)
    gdf_gye = gdf_shp[gdf_shp['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
    if gdf_gye.crs is None:
        gdf_gye = gdf_gye.set_crs("EPSG:32717")
    
    # Dissolve by parish name
    gdf_parr = gdf_gye.dissolve(by='DPA_DESPAR', as_index=False)
    # Convert parish GeoDataFrame to same CRS as gdf_total (EPSG:4326)
    gdf_parr = gdf_parr.to_crs(gdf_total.crs)
    
    # Load trace and compute RR median
    import arviz as az
    trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    trace = az.from_netcdf(trace_path)
    
    alpha_samples = trace.posterior['alpha'].values.flatten()
    phi_samples = trace.posterior['phi'].values.reshape(-1, trace.posterior['phi'].shape[-1])
    n_samples, n_zonas = phi_samples.shape
    
    rr_samples = np.zeros((n_samples, n_zonas))
    for s in range(n_samples):
        rr_samples[s, :] = np.exp(alpha_samples[s] + phi_samples[s, :])
        
    rr_median = np.median(rr_samples, axis=0)
    
    sub_codes = sorted(gdf_total['cod_subcircu'].unique())
    df_rr = pd.DataFrame({
        'cod_subcircu': sub_codes,
        'rr_estimado_mediana': rr_median
    })
    
    gdf_rr = gdf_total.merge(df_rr, on='cod_subcircu', how='left')
    
    # Classify RR
    bins = [0, 0.8, 1.0, 1.2, 1.5, np.inf]
    labels = [
        'Riesgo Bajo (RR < 0.8)',
        'Riesgo Moderado-Bajo (0.8 - 1.0)',
        'Riesgo Moderado-Alto (1.0 - 1.2)',
        'Riesgo Alto (1.2 - 1.5)',
        'Riesgo Muy Alto (RR > 1.5)'
    ]
    gdf_rr['rr_categoria'] = pd.cut(gdf_rr['rr_estimado_mediana'], bins=bins, labels=labels)
    
    # Colors
    colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Plot subcircuits colored by RR
    gdf_rr.plot(
        column='rr_categoria',
        cmap=cmap,
        legend=True,
        legend_kwds={
            'title': 'Riesgo Relativo Posterior (RR)',
            'loc': 'lower left',
            'fontsize': 10,
            'title_fontsize': 11,
            'frameon': True,
            'facecolor': 'white',
            'edgecolor': '#cccccc'
        },
        ax=ax,
        edgecolor='#aaaaaa',
        linewidth=0.2,
        alpha=0.95
    )
    
    # 2. Plot parish boundaries (dissolved)
    gdf_parr.plot(
        ax=ax,
        facecolor='none',
        edgecolor='black',
        linewidth=1.2,
        linestyle='--',
        zorder=3
    )
    
    # 3. Annotate parishes at their centroids
    # Adjust some text placement to avoid overlap or out of bounds
    for idx, row in gdf_parr.iterrows():
        name = row['DPA_DESPAR'].split('(')[0].strip() # Clean name, e.g. "JUAN GOMEZ RENDON"
        if "GUAYAQUIL" in name:
            name = "GUAYAQUIL (URBANO)"
        centroid = row.geometry.centroid
        
        # Custom adjustments for better label positioning
        cx, cy = centroid.x, centroid.y
        if "PUNA" in name:
            cy += 0.05
        elif "MORRO" in name:
            cx -= 0.02
        elif "POSORJA" in name:
            cy -= 0.02
            
        ax.text(
            cx, cy, name,
            fontsize=9,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8),
            zorder=4
        )
        
    ax.set_title("Mapa de Riesgo Relativo Bayesiano (Clasificado)\nUnidades Espaciales (Subcircuitos) y Parroquias de Guayaquil", fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    out_path = os.path.join(config.FIGURES_DIR, "test_parish_overlay.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa de parroquias guardado en: {out_path}")

if __name__ == "__main__":
    main()
