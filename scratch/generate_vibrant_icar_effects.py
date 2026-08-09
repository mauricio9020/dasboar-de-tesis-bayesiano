import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patches as patches
import arviz as az

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def main():
    print("Cargando datos y traza bayesiana...")
    # Load subcircuits GeoJSON
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
    gdf_total = gpd.read_file(geojson_path)
    
    # Load PyMC trace
    trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    if not os.path.exists(trace_path):
        print(f"Error: No existe la traza en {trace_path}")
        return
    trace = az.from_netcdf(trace_path)
    
    # Extract posterior samples of structured spatial effect 'phi'
    phi_samples = trace.posterior['phi'].values.reshape(-1, trace.posterior['phi'].shape[-1])
    phi_median = np.median(phi_samples, axis=0)
    
    # Create DataFrame
    sub_codes = sorted(gdf_total['cod_subcircu'].unique())
    df_phi = pd.DataFrame({
        'cod_subcircu': sub_codes,
        'efecto_espacial_mediana': phi_median
    })
    
    gdf_phi_map = gdf_total.merge(df_phi, on='cod_subcircu', how='left')
    gdf_phi_map_utm = gdf_phi_map.to_crs("EPSG:32717")
    
    # Load parish boundaries
    gdf_parr_utm = None
    if os.path.exists(config.SHAPEFILE_PATH):
        try:
            gdf_shp = gpd.read_file(config.SHAPEFILE_PATH)
            gdf_gye_shp = gdf_shp[gdf_shp['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
            if gdf_gye_shp.crs is None:
                gdf_gye_shp = gdf_gye_shp.set_crs("EPSG:32717")
            gdf_parr = gdf_gye_shp.dissolve(by='DPA_DESPAR', as_index=False)
            gdf_parr_utm = gdf_parr.to_crs("EPSG:32717")
        except Exception as e:
            print(f"Advertencia: No se pudo cargar o disolver parroquias: {e}")
            
    # Create custom, highly vibrant diverging colormap
    # From deep royal blue -> sky blue -> off-white -> coral red -> deep crimson red
    colors = ["#0d47a1", "#2196f3", "#f8f9fa", "#ff5252", "#b71c1c"]
    vibrant_cmap = LinearSegmentedColormap.from_list("vibrant_rdbu", colors)
    
    # Symmetric normalization around 0
    max_abs_val = max(abs(phi_median))
    norm = Normalize(vmin=-max_abs_val, vmax=max_abs_val)
    
    # Set up 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11), facecolor='white')
    
    # -------------------------------------------------------------------------
    # PANEL A: MAPA CANTONAL COMPLETO (RURAL E INSULAR)
    # -------------------------------------------------------------------------
    print("Graficando Panel A (Cantonal)...")
    gdf_phi_map_utm.plot(
        column='efecto_espacial_mediana',
        cmap=vibrant_cmap,
        norm=norm,
        ax=ax1,
        edgecolor='#777777',
        linewidth=0.2,
        alpha=0.95
    )
    
    if gdf_parr_utm is not None:
        gdf_parr_utm.plot(
            ax=ax1,
            facecolor='none',
            edgecolor='#111111',
            linewidth=1.2,
            linestyle='--',
            zorder=3
        )
        
        # Label parishes on Panel A
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
                
            ax1.text(
                cx, cy, name,
                fontsize=8,
                fontweight='bold',
                color='black',
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", lw=0.5, alpha=0.8),
                zorder=4
            )
            
    # Set limits for Panel A
    xmin, ymin, xmax, ymax = gdf_phi_map_utm.total_bounds
    x_pad = (xmax - xmin) * 0.02
    y_pad = (ymax - ymin) * 0.02
    ax1.set_xlim(xmin - x_pad, xmax + x_pad)
    ax1.set_ylim(ymin - y_pad, ymax + y_pad)
    
    # Scale Bar (10 km) on Panel A
    scale_len = 10000
    scale_x_start = xmax - scale_len - 5000
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
    ax1.add_patch(scale_box)
    ax1.plot([scale_x_start, scale_x_start + scale_len], [scale_y, scale_y], color='black', linewidth=2.5, zorder=5)
    ax1.plot([scale_x_start, scale_x_start], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.2, zorder=5)
    ax1.plot([scale_x_start + scale_len/2, scale_x_start + scale_len/2], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.2, zorder=5)
    ax1.plot([scale_x_start + scale_len, scale_x_start + scale_len], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.2, zorder=5)
    
    ax1.text(scale_x_start, scale_y + text_offset, '0', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    ax1.text(scale_x_start + scale_len/2, scale_y + text_offset, '5 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    ax1.text(scale_x_start + scale_len, scale_y + text_offset, '10 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    
    # North Arrow on Panel A
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
    ax1.add_patch(rect_n)
    ax1.annotate(
        'N', xy=(arrow_x, arrow_y + 4000), xytext=(arrow_x, arrow_y - 4000),
        arrowprops=dict(facecolor='black', width=3, headwidth=10, shrink=0.05),
        ha='center', va='center', fontsize=14, fontweight='bold', color='black',
        zorder=5
    )
    
    rect_border1 = patches.Rectangle(
        (xmin - x_pad/2, ymin - y_pad/2),
        (xmax - xmin) + x_pad,
        (ymax - ymin) + y_pad,
        linewidth=1,
        edgecolor='#aaaaaa',
        facecolor='none',
        zorder=10
    )
    ax1.add_patch(rect_border1)
    ax1.set_title("A: Extensión Cantonal Completa", fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # -------------------------------------------------------------------------
    # PANEL B: DETALLE DEL ÁREA URBANA CONSOLIDADA
    # -------------------------------------------------------------------------
    print("Graficando Panel B (Zoom Urbano)...")
    gdf_phi_map_utm.plot(
        column='efecto_espacial_mediana',
        cmap=vibrant_cmap,
        norm=norm,
        ax=ax2,
        edgecolor='#555555',
        linewidth=0.4,
        alpha=0.95
    )
    
    if gdf_parr_utm is not None:
        gdf_parr_utm.plot(
            ax=ax2,
            facecolor='none',
            edgecolor='#111111',
            linewidth=1.8,
            linestyle='--',
            zorder=3
        )
        
    # Set urban coordinates limits: xmin=600949.5, ymin=9748118.8, xmax=626091.7, ymax=9783094.4
    xmin_urb, ymin_urb, xmax_urb, ymax_urb = 600949.5, 9748118.8, 626091.7, 9783094.4
    x_pad_urb = (xmax_urb - xmin_urb) * 0.05
    y_pad_urb = (ymax_urb - ymin_urb) * 0.05
    ax2.set_xlim(xmin_urb - x_pad_urb, xmax_urb + x_pad_urb)
    ax2.set_ylim(ymin_urb - y_pad_urb, ymax_urb + y_pad_urb)
    
    # Label top 5 critical subcircuits directly on Panel B
    top_5_names = ['TRINITARIA NORTE 4', 'SUBURBIO 1', 'GUASMO 1', 'NUEVO GUAYAQUIL 3', 'PARAISO LA FLOR 4']
    for idx, row in gdf_phi_map_utm.iterrows():
        name = row['subcircuito_nom']
        if name in top_5_names:
            centroid = row.geometry.centroid
            # Shift a bit to avoid overlaps if necessary
            cx_u, cy_u = centroid.x, centroid.y
            if name == 'TRINITARIA NORTE 4':
                cy_u += 300
            elif name == 'SUBURBIO 1':
                cx_u -= 400
                
            ax2.text(
                cx_u, cy_u, name,
                fontsize=7,
                fontweight='bold',
                color='black',
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="#d32f2f", lw=0.8, alpha=0.9),
                zorder=5
            )
            
    # Add a scale bar for the zoom (5 km)
    scale_len_b = 5000
    scale_x_start_b = xmax_urb - scale_len_b - 2000
    scale_y_b = ymin_urb + 2000
    tick_height_b = (ymax_urb - ymin_urb) * 0.008
    text_offset_b = tick_height_b * 1.5
    
    scale_box_b = patches.Rectangle(
        (scale_x_start_b - 1000, scale_y_b - tick_height_b * 2),
        scale_len_b + 2000,
        tick_height_b * 5,
        linewidth=0.5,
        edgecolor='#cccccc',
        facecolor='white',
        alpha=0.85,
        zorder=4
    )
    ax2.add_patch(scale_box_b)
    ax2.plot([scale_x_start_b, scale_x_start_b + scale_len_b], [scale_y_b, scale_y_b], color='black', linewidth=2.5, zorder=5)
    ax2.plot([scale_x_start_b, scale_x_start_b], [scale_y_b - tick_height_b/2, scale_y_b + tick_height_b/2], color='black', linewidth=1.2, zorder=5)
    ax2.plot([scale_x_start_b + scale_len_b/2, scale_x_start_b + scale_len_b/2], [scale_y_b - tick_height_b/2, scale_y_b + tick_height_b/2], color='black', linewidth=1.2, zorder=5)
    ax2.plot([scale_x_start_b + scale_len_b, scale_x_start_b + scale_len_b], [scale_y_b - tick_height_b/2, scale_y_b + tick_height_b/2], color='black', linewidth=1.2, zorder=5)
    
    ax2.text(scale_x_start_b, scale_y_b + text_offset_b, '0', ha='center', fontsize=8, fontweight='bold', color='black', zorder=6)
    ax2.text(scale_x_start_b + scale_len_b/2, scale_y_b + text_offset_b, '2.5 km', ha='center', fontsize=8, fontweight='bold', color='black', zorder=6)
    ax2.text(scale_x_start_b + scale_len_b, scale_y_b + text_offset_b, '5 km', ha='center', fontsize=8, fontweight='bold', color='black', zorder=6)
    
    rect_border2 = patches.Rectangle(
        (xmin_urb - x_pad_urb/2, ymin_urb - y_pad_urb/2),
        (xmax_urb - xmin_urb) + x_pad_urb,
        (ymax_urb - ymin_urb) + y_pad_urb,
        linewidth=1,
        edgecolor='#aaaaaa',
        facecolor='none',
        zorder=10
    )
    ax2.add_patch(rect_border2)
    ax2.set_title("B: Detalle del Área Urbana Consolidada", fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')
    
    # -------------------------------------------------------------------------
    # COLORBAR COMPARTIDA Y DETALLES FINALES
    # -------------------------------------------------------------------------
    # Add colorbar on the right side
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap=vibrant_cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Efecto Espacial Estructurado ($\phi_i$ mediana posterior)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Main super title
    plt.suptitle("Distribución espacial del efecto ICAR estimado por el modelo bayesiano\nCantón Guayaquil: Parroquias y Subcircuitos", fontsize=17, fontweight='bold', y=0.98)
    
    # Save files
    png_path = os.path.join(config.FIGURES_DIR, "mapa_efectos_espaciales_icar.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "mapa_efectos_espaciales_icar.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] Mapa de efectos espaciales vibrante de 2 paneles guardado en PNG: {png_path}")

if __name__ == "__main__":
    main()
