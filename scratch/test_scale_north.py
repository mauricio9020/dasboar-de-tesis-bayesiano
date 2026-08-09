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
    
    # Project both to UTM Zone 17S (EPSG:32717)
    gdf_total_utm = gdf_total.to_crs("EPSG:32717")
    gdf_parr_utm = gdf_parr.to_crs("EPSG:32717")
    
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
    
    sub_codes = sorted(gdf_total_utm['cod_subcircu'].unique())
    df_rr = pd.DataFrame({
        'cod_subcircu': sub_codes,
        'rr_estimado_mediana': rr_median
    })
    
    gdf_rr_utm = gdf_total_utm.merge(df_rr, on='cod_subcircu', how='left')
    
    # Classify RR
    bins = [0, 0.8, 1.0, 1.2, 1.5, np.inf]
    labels = [
        'Riesgo Bajo (RR < 0.8)',
        'Riesgo Moderado-Bajo (0.8 - 1.0)',
        'Riesgo Moderado-Alto (1.0 - 1.2)',
        'Riesgo Alto (1.2 - 1.5)',
        'Riesgo Muy Alto (RR > 1.5)'
    ]
    gdf_rr_utm['rr_categoria'] = pd.cut(gdf_rr_utm['rr_estimado_mediana'], bins=bins, labels=labels)
    
    # Colors
    colors = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # 1. Plot subcircuits colored by RR
    gdf_rr_utm.plot(
        column='rr_categoria',
        cmap=cmap,
        legend=True,
        legend_kwds={
            'title': 'Riesgo Relativo Posterior (RR)',
            'loc': 'lower left',
            'fontsize': 11,
            'title_fontsize': 12,
            'frameon': True,
            'facecolor': 'white',
            'edgecolor': '#cccccc',
            'shadow': False
        },
        ax=ax,
        edgecolor='#666666',
        linewidth=0.2,
        alpha=0.95
    )
    
    # 2. Plot parish boundaries (dissolved)
    gdf_parr_utm.plot(
        ax=ax,
        facecolor='none',
        edgecolor='#111111',
        linewidth=1.3,
        linestyle='--',
        zorder=3
    )
    
    # 3. Annotate parishes at their centroids
    for idx, row in gdf_parr_utm.iterrows():
        name = row['DPA_DESPAR'].split('(')[0].strip() # Clean name, e.g. "JUAN GOMEZ RENDON"
        if "GUAYAQUIL" in name:
            name = "GUAYAQUIL (URBANO)"
        centroid = row.geometry.centroid
        
        # Custom adjustments for better label positioning in UTM
        cx, cy = centroid.x, centroid.y
        if "PUNA" in name:
            cy += 5000
        elif "MORRO" in name:
            cx -= 2000
        elif "POSORJA" in name:
            cy -= 2000
            
        ax.text(
            cx, cy, name,
            fontsize=9,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", lw=0.5, alpha=0.85),
            zorder=4
        )
        
    # Get bounds
    xmin, ymin, xmax, ymax = gdf_rr_utm.total_bounds
    x_pad = (xmax - xmin) * 0.02
    y_pad = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    
    # 4. Draw Scale Bar (10 km) in the bottom-right (above Tenguel, around x = 620,000, y = 9,700,000)
    # Let's place it at x = xmax - 25000, y = ymin + 15000
    scale_len = 10000  # meters (10 km)
    scale_x_start = xmax - scale_len - 10000
    scale_y = ymin + 15000
    
    tick_height = (ymax - ymin) * 0.008
    text_offset = tick_height * 1.5
    
    # Scale background box
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
    ax.add_patch(scale_box)
    
    # Draw scale lines
    ax.plot([scale_x_start, scale_x_start + scale_len], [scale_y, scale_y], color='black', linewidth=2.5, zorder=5)
    ax.plot([scale_x_start, scale_x_start], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
    ax.plot([scale_x_start + scale_len/2, scale_x_start + scale_len/2], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
    ax.plot([scale_x_start + scale_len, scale_x_start + scale_len], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
    
    # Scale labels
    ax.text(scale_x_start, scale_y + text_offset, '0', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    ax.text(scale_x_start + scale_len/2, scale_y + text_offset, '5 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    ax.text(scale_x_start + scale_len, scale_y + text_offset, '10 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    
    # 5. Draw North Arrow in top-left
    # Let's put a nice compass-like arrow
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
    ax.add_patch(rect_n)
    
    # Draw the north arrow
    ax.annotate(
        'N', xy=(arrow_x, arrow_y + 4000), xytext=(arrow_x, arrow_y - 4000),
        arrowprops=dict(facecolor='black', width=3, headwidth=10, shrink=0.05),
        ha='center', va='center', fontsize=14, fontweight='bold', color='black',
        zorder=5
    )
    
    ax.set_title("MAPA DE RIESGO RELATIVO BAYESIANO (CLASIFICADO)\nCanton Guayaquil: Parroquias y Subcircuitos", fontsize=15, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add a border around the map plot area to make it look clean
    rect_border = patches.Rectangle(
        (xmin - x_pad/2, ymin - y_pad/2),
        (xmax - xmin) + x_pad,
        (ymax - ymin) + y_pad,
        linewidth=1,
        edgecolor='#aaaaaa',
        facecolor='none',
        zorder=10
    )
    ax.add_patch(rect_border)
    
    out_path = os.path.join(config.FIGURES_DIR, "test_scale_north.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa final de prueba guardado en: {out_path}")

if __name__ == "__main__":
    main()
