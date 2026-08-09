import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import arviz as az

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def main():
    print("Cargando datos...")
    # Load subcircuits GeoJSON
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
    gdf_total = gpd.read_file(geojson_path)
    
    # Load PyMC trace
    trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    trace = az.from_netcdf(trace_path)
    
    # Extract posterior spatial effects and calculate relative risk
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
    
    # Classify RR by Quintiles
    q = gdf_rr['rr_estimado_mediana'].quantile([0.2, 0.4, 0.6, 0.8]).values
    labels = [
        f'Muy Bajo (RR ≤ {q[0]:.2f})',
        f'Bajo ({q[0]:.2f} - {q[1]:.2f})',
        f'Medio ({q[1]:.2f} - {q[2]:.2f})',
        f'Alto ({q[2]:.2f} - {q[3]:.2f})',
        f'Muy Alto (RR > {q[3]:.2f})'
    ]
    gdf_rr['rr_categoria'] = pd.qcut(gdf_rr['rr_estimado_mediana'], q=5, labels=labels)
    
    # Project to Web Mercator (EPSG:3857) for Contextily basemap
    gdf_rr_3857 = gdf_rr.to_crs(epsg=3857)
    gdf_rr_3857['cx'] = gdf_rr_3857.geometry.centroid.x
    gdf_rr_3857['cy'] = gdf_rr_3857.geometry.centroid.y
    
    # Load parish boundaries
    gdf_parr_3857 = None
    if os.path.exists(config.SHAPEFILE_PATH):
        try:
            gdf_shp = gpd.read_file(config.SHAPEFILE_PATH)
            gdf_gye_shp = gdf_shp[gdf_shp['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
            if gdf_gye_shp.crs is None:
                gdf_gye_shp = gdf_gye_shp.set_crs("EPSG:32717")
            gdf_parr = gdf_gye_shp.dissolve(by='DPA_DESPAR', as_index=False)
            gdf_parr_3857 = gdf_parr.to_crs(epsg=3857)
        except Exception as e:
            print(f"Error cargando parroquias: {e}")

    # Set up dark-themed figure
    fig, ax = plt.subplots(figsize=(12, 13), facecolor='#111111')
    ax.set_facecolor('#111111')
    
    # Plot parish boundaries
    if gdf_parr_3857 is not None:
        gdf_parr_3857.plot(
            ax=ax,
            facecolor='none',
            edgecolor='#555555',
            linewidth=1.0,
            linestyle='--',
            alpha=0.6,
            zorder=2
        )
        
        # Annotate parishes in dark style
        for idx, row in gdf_parr_3857.iterrows():
            name = row['DPA_DESPAR'].split('(')[0].strip()
            if "GUAYAQUIL" in name:
                name = "GUAYAQUIL (URBANO)"
            centroid = row.geometry.centroid
            cx_coord, cy_coord = centroid.x, centroid.y
            if "PUNA" in name:
                cy_coord += 5000
            elif "MORRO" in name:
                cx_coord -= 2000
            elif "POSORJA" in name:
                cy_coord -= 2000
                
            ax.text(
                cx_coord, cy_coord, name,
                fontsize=8,
                fontweight='bold',
                color='#ffffff',
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="#181818", ec="#444444", lw=0.5, alpha=0.8),
                zorder=4
            )
            
    # Map colors and sizes to categories
    cats = gdf_rr_3857['rr_categoria'].cat.categories
    cat_colors = {
        cats[0]: '#2ecc71',  # Emerald Green (Very Low)
        cats[1]: '#27ae60',  # Green (Low)
        cats[2]: '#f1c40f',  # Yellow (Medium)
        cats[3]: '#e67e22',  # Orange (High)
        cats[4]: '#e74c3c'   # Alizarin Red (Very High)
    }
    cat_sizes = {
        cats[0]: 20,
        cats[1]: 35,
        cats[2]: 70,
        cats[3]: 140,
        cats[4]: 280
    }
    
    # Plot bubbles for each category
    for cat in cats:
        sub_gdf = gdf_rr_3857[gdf_rr_3857['rr_categoria'] == cat]
        alpha = 0.55 if cat in [cats[3], cats[4]] else 0.8
        ax.scatter(
            sub_gdf['cx'], sub_gdf['cy'],
            s=cat_sizes[cat],
            color=cat_colors[cat],
            label=cat,
            alpha=alpha,
            edgecolors='#111111' if cat in [cats[3], cats[4]] else 'none',
            linewidths=0.5,
            zorder=3
        )
        
    # Set boundaries
    xmin, ymin, xmax, ymax = gdf_rr_3857.total_bounds
    x_pad = (xmax - xmin) * 0.05
    y_pad = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    
    # Plot basemap (CartoDB DarkMatter) - now axes limits are correctly set!
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, alpha=0.9, zorder=1)
    
    # Legend in dark style
    legend = ax.legend(
        title="Categorías de Riesgo Relativo (RR)",
        loc="lower left",
        facecolor="#181818",
        edgecolor="#444444",
        labelcolor="#ffffff",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.9
    )
    legend.get_title().set_color('#ffffff')
    
    # Scale Bar (10 km) in dark style
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
        edgecolor='#444444',
        facecolor='#181818',
        alpha=0.85,
        zorder=4
    )
    ax.add_patch(scale_box)
    
    ax.plot([scale_x_start, scale_x_start + scale_len], [scale_y, scale_y], color='#ffffff', linewidth=2.0, zorder=5)
    ax.plot([scale_x_start, scale_x_start], [scale_y - tick_height/2, scale_y + tick_height/2], color='#ffffff', linewidth=1.2, zorder=5)
    ax.plot([scale_x_start + scale_len/2, scale_x_start + scale_len/2], [scale_y - tick_height/2, scale_y + tick_height/2], color='#ffffff', linewidth=1.2, zorder=5)
    ax.plot([scale_x_start + scale_len, scale_x_start + scale_len], [scale_y - tick_height/2, scale_y + tick_height/2], color='#ffffff', linewidth=1.2, zorder=5)
    
    ax.text(scale_x_start, scale_y + text_offset, '0', ha='center', fontsize=9, fontweight='bold', color='#ffffff', zorder=6)
    ax.text(scale_x_start + scale_len/2, scale_y + text_offset, '5 km', ha='center', fontsize=9, fontweight='bold', color='#ffffff', zorder=6)
    ax.text(scale_x_start + scale_len, scale_y + text_offset, '10 km', ha='center', fontsize=9, fontweight='bold', color='#ffffff', zorder=6)
    
    # North Arrow in dark style
    arrow_x = xmin + 15000
    arrow_y = ymax - 20000
    rect_n = patches.Rectangle(
        (arrow_x - 5000, arrow_y - 7000),
        10000,
        14000,
        linewidth=0.5,
        edgecolor='#444444',
        facecolor='#181818',
        alpha=0.85,
        zorder=4
    )
    ax.add_patch(rect_n)
    ax.annotate(
        'N', xy=(arrow_x, arrow_y + 4000), xytext=(arrow_x, arrow_y - 4000),
        arrowprops=dict(facecolor='#ffffff', width=3, headwidth=10, shrink=0.05),
        ha='center', va='center', fontsize=14, fontweight='bold', color='#ffffff',
        zorder=5
    )
    
    # Aesthetic frame
    rect_border = patches.Rectangle(
        (xmin - x_pad/2, ymin - y_pad/2),
        (xmax - xmin) + x_pad,
        (ymax - ymin) + y_pad,
        linewidth=1,
        edgecolor='#444444',
        facecolor='none',
        zorder=10
    )
    ax.add_patch(rect_border)
    
    ax.set_title("MAPA DE BURBUJAS DE RIESGO RELATIVO BAYESIANO (TEMÁTICO)\nCantón Guayaquil: Parroquias y Subcircuitos", fontsize=14, fontweight='bold', color='#ffffff', pad=20)
    ax.axis('off')
    
    out_path = os.path.join(config.FIGURES_DIR, "mapa_burbujas_riesgo_oscuro.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"Mapa de burbujas guardado con éxito en: {out_path}")

if __name__ == "__main__":
    main()
