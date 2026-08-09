import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
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
    # Compute median posterior for each subcircuit
    phi_median = np.median(phi_samples, axis=0)
    phi_mean = np.mean(phi_samples, axis=0)
    phi_lower = np.percentile(phi_samples, 2.5, axis=0)
    phi_upper = np.percentile(phi_samples, 97.5, axis=0)
    
    # Create DataFrame
    sub_codes = sorted(gdf_total['cod_subcircu'].unique())
    df_phi = pd.DataFrame({
        'cod_subcircu': sub_codes,
        'efecto_espacial_mediana': phi_median,
        'efecto_espacial_media': phi_mean,
        'efecto_espacial_inf_95': phi_lower,
        'efecto_espacial_sup_95': phi_upper
    })
    
    # Merge administrative names
    df_names = gdf_total[['cod_subcircu', 'subcircuito_nom', 'circuito_nom', 'distrito_nom', 'homicidios_totales']].drop_duplicates(subset=['cod_subcircu'])
    df_phi_complete = df_names.merge(df_phi, on='cod_subcircu', how='right').sort_values(by='efecto_espacial_mediana', ascending=False)
    
    # Save CSV and Excel
    csv_out_path = os.path.join(config.CLEAN_DATA_DIR, "efectos_espaciales_icar.csv")
    excel_out_path = os.path.join(config.TABLES_DIR, "tabla_efectos_espaciales_icar.xlsx")
    
    df_phi_complete.to_csv(csv_out_path, index=False)
    df_phi_complete.to_excel(excel_out_path, index=False)
    print(f"[OK] Tabla de efectos guardada en CSV: {csv_out_path}")
    print(f"[OK] Tabla de efectos guardada en Excel: {excel_out_path}")
    
    # Generate Markdown Table Artifact
    md_table = """# Tabla Completa de Efectos Espaciales ICAR (phi) por Subcircuito

Esta tabla muestra los efectos espaciales estructurados ($\phi_i$) del prior condicional autorregresivo (ICAR) estimado por el modelo. 
Un valor **positivo** indica que las condiciones geográficas locales aumentan el riesgo en la zona, mientras que un valor **negativo** representa un efecto protector o de reducción del riesgo.

### 📊 Tabla de Efectos Espaciales Estructurados (phi)

| N° | Subcircuito | Código | Distrito | Circuito | Homicidios Totales | Efecto Espacial (Mediana) | Intervalo de Credibilidad (95%) |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: |
"""
    for idx, row in df_phi_complete.iterrows():
        name = row['subcircuito_nom']
        code = row['cod_subcircu']
        distrito = row['distrito_nom']
        circuito = row['circuito_nom']
        homicidios = int(row['homicidios_totales']) if pd.notnull(row['homicidios_totales']) else 0
        val = row['efecto_espacial_mediana']
        lower = row['efecto_espacial_inf_95']
        upper = row['efecto_espacial_sup_95']
        
        md_table += f"| {idx+1} | {name} | `{code}` | {distrito} | {circuito} | {homicidios} | **{val:+.4f}** | [{lower:+.4f}, {upper:+.4f}] |\n"
        
    artifact_table_path = r"C:\Users\ASUS\\.gemini\antigravity-ide\brain\8000dc29-2b20-4530-8269-0f2ed3f93f26\tabla_efectos_espaciales_completa.md"
    with open(artifact_table_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"[OK] Tabla markdown guardada en: {artifact_table_path}")
    
    # -------------------------------------------------------------------------
    # GENERACIÓN DE MAPA DE EFECTOS ESPACIALES (RdBu Diverging Colormap)
    # -------------------------------------------------------------------------
    print("Generando mapa de efectos espaciales ICAR...")
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
            gdf_parr_utm = gdf_gye_shp.dissolve(by='DPA_DESPAR', as_index=False)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar o disolver parroquias: {e}")
            
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Normalize colormap symmetrically around 0.0
    # Red = positive effect, Blue = negative effect, White = neutral
    max_abs_val = max(abs(df_phi_complete['efecto_espacial_mediana']))
    # Avoid zero boundary error
    max_abs_val = max_abs_val if max_abs_val > 0 else 1.0
    norm = Normalize(vmin=-max_abs_val, vmax=max_abs_val)
    
    # Plot subcircuits colored by structured spatial effect phi (using RdBu_r so red is positive and blue is negative)
    gdf_phi_map_utm.plot(
        column='efecto_espacial_mediana',
        cmap='RdBu_r',
        norm=norm,
        legend=True,
        legend_kwds={
            'label': 'Efecto Espacial Estructurado ($\phi_i$ mediana posterior)',
            'orientation': 'vertical',
            'shrink': 0.6,
            'pad': 0.02
        },
        ax=ax,
        edgecolor='#666666',
        linewidth=0.2,
        alpha=0.95
    )
    
    # Draw parish boundaries
    if gdf_parr_utm is not None:
        gdf_parr_utm.plot(
            ax=ax,
            facecolor='none',
            edgecolor='#111111',
            linewidth=1.3,
            linestyle='--',
            zorder=3
        )
        
        # Annotations of parishes at centroids
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
            
    # Set plot boundaries
    xmin, ymin, xmax, ymax = gdf_phi_map_utm.total_bounds
    x_pad = (xmax - xmin) * 0.02
    y_pad = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    
    # Draw Scale Bar (10 km)
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
    ax.add_patch(scale_box)
    
    ax.plot([scale_x_start, scale_x_start + scale_len], [scale_y, scale_y], color='black', linewidth=2.5, zorder=5)
    ax.plot([scale_x_start, scale_x_start], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
    ax.plot([scale_x_start + scale_len/2, scale_x_start + scale_len/2], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
    ax.plot([scale_x_start + scale_len, scale_x_start + scale_len], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
    
    ax.text(scale_x_start, scale_y + text_offset, '0', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    ax.text(scale_x_start + scale_len/2, scale_y + text_offset, '5 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    ax.text(scale_x_start + scale_len, scale_y + text_offset, '10 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
    
    # Draw North Arrow
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
    ax.annotate(
        'N', xy=(arrow_x, arrow_y + 4000), xytext=(arrow_x, arrow_y - 4000),
        arrowprops=dict(facecolor='black', width=3, headwidth=10, shrink=0.05),
        ha='center', va='center', fontsize=14, fontweight='bold', color='black',
        zorder=5
    )
    
    # Outer Border Rectangle
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
    
    ax.set_title("Distribución espacial del efecto ICAR estimado por el modelo bayesiano\nCantón Guayaquil: Parroquias y Subcircuitos", fontsize=15, fontweight='bold', pad=20)
    ax.axis('off')
    
    png_path = os.path.join(config.FIGURES_DIR, "mapa_efectos_espaciales_icar.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "mapa_efectos_espaciales_icar.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] Mapa de efectos espaciales guardado en PNG: {png_path}")

if __name__ == "__main__":
    main()
