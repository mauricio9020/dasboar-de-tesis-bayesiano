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
    # Load GeoJSON
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
    if not os.path.exists(geojson_path):
        print(f"Error: No existe {geojson_path}")
        return
        
    gdf_total = gpd.read_file(geojson_path)
    
    # Load PyMC trace
    import arviz as az
    trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    if not os.path.exists(trace_path):
        print(f"Error: No existe la traza en {trace_path}")
        return
        
    print("Cargando traza posterior...")
    trace = az.from_netcdf(trace_path)
    
    # Extract posterior samples
    alpha_samples = trace.posterior['alpha'].values.flatten()
    phi_samples = trace.posterior['phi'].values.reshape(-1, trace.posterior['phi'].shape[-1])
    n_samples, n_zonas = phi_samples.shape
    
    # Compute RR
    rr_samples = np.zeros((n_samples, n_zonas))
    for s in range(n_samples):
        rr_samples[s, :] = np.exp(alpha_samples[s] + phi_samples[s, :])
        
    rr_median = np.median(rr_samples, axis=0)
    
    # Map results
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
    
    # Define colors matching a diverging palette: Blue/Green -> Yellow -> Orange -> Red
    colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot gdf
    gdf_rr.plot(
        column='rr_categoria',
        cmap=cmap,
        legend=True,
        legend_kwds={
            'title': 'Categorías de Riesgo Relativo (RR)',
            'loc': 'lower left',
            'fontsize': 10,
            'title_fontsize': 11,
            'frameon': True,
            'facecolor': 'white',
            'edgecolor': '#cccccc'
        },
        ax=ax,
        edgecolor='#444444',
        linewidth=0.3
    )
    
    ax.set_title("Mapa de Riesgo Relativo Bayesiano (Clasificado)\nDistritos y Subcircuitos de Guayaquil", fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    out_path = os.path.join(config.FIGURES_DIR, "test_rr_map_classified.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa de prueba guardado en: {out_path}")

if __name__ == "__main__":
    main()
