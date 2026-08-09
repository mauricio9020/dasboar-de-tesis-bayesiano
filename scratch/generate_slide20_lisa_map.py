import os
import sys
import shutil
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Artifact directory for embedding images
ARTIFACT_DIR = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\8c0ff45f-78e7-412d-8e2e-8062467c5b75"

def generate_slide20_lisa_map():
    print("Cargando datos geográficos ESDA...")
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_esda.geojson")
    if not os.path.exists(geojson_path):
        geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
        
    if not os.path.exists(geojson_path):
        print(f"Error: No se encontró el GeoJSON en {geojson_path}")
        return
        
    gdf = gpd.read_file(geojson_path)
    
    # Proyectar a UTM 17S (EPSG:32717) o mantener Web Mercator (EPSG:3857)
    # Usaremos UTM 17S para proporciones perfectas
    if gdf.crs != "EPSG:32717":
        gdf = gdf.to_crs("EPSG:32717")
        
    # 1. Filtrar áreas urbanas (excluyendo distritos rurales distantes)
    gdf_urban = gdf[
        (~gdf['subcircuito_nom'].str.upper().str.contains('PUNA|TENGUEL|MORRO|POSORJA|PROGRESO', na=False)) &
        (~gdf['distrito_nom'].str.upper().str.contains('PROGRESO|PLAYAS|SAMBORONDON|DAULE|NARANJAL', na=False))
    ].copy()
    
    print(f"Total subcircuitos urbanos: {len(gdf_urban)}")
    
    # 2. Definir paleta de colores premium para LISA
    colors_dict = {
        'HIGH-HIGH (HOTSPOT)': '#ef4444',  # Rojo Coral (Foco caliente)
        'LOW-LOW (COLDSPOT)': '#3b82f6',   # Azul Muted (Foco frío)
        'HIGH-LOW (OUTLIER)': '#f59e0b',   # Naranja / Ámbar (Alto rodeado de Bajo)
        'LOW-HIGH (OUTLIER)': '#c084fc',   # Lavanda / Púrpura claro (Bajo rodeado de Alto)
        'NOT SIGNIFICANT': '#f3f4f6'       # Gris claro neutro
    }
    
    # Mapear nombres en español para la leyenda
    legend_labels = {
        'HIGH-HIGH (HOTSPOT)': 'Alto-Alto (Hotspot de Alta Violencia)',
        'LOW-LOW (COLDSPOT)': 'Bajo-Bajo (Coldspot / Baja Violencia)',
        'HIGH-LOW (OUTLIER)': 'Alto-Bajo (Outlier Espacial)',
        'LOW-HIGH (OUTLIER)': 'Bajo-Alto (Outlier Espacial)',
        'NOT SIGNIFICANT': 'No Significativo'
    }
    
    # 3. Calcular centroides específicos para apuntar las anotaciones
    # Filtramos por clústeres High-High en los distritos de interés
    hh_gdf = gdf_urban[gdf_urban['lisa_category'] == 'HIGH-HIGH (HOTSPOT)']
    
    # Nueva Prosperina
    nva_prop_hh = hh_gdf[hh_gdf['distrito_nom'].str.upper() == 'NUEVA PROSPERINA']
    c_nva_prop = nva_prop_hh.geometry.union_all().centroid
    x_nva, y_nva = c_nva_prop.x, c_nva_prop.y
        
    # Sur / Guasmo
    sur_hh = hh_gdf[hh_gdf['distrito_nom'].str.upper() == 'SUR']
    c_sur = sur_hh.geometry.union_all().centroid
    x_sur, y_sur = c_sur.x, c_sur.y
        
    # Pascuales
    pas_hh = hh_gdf[hh_gdf['distrito_nom'].str.upper() == 'PASCUALES']
    c_pas = pas_hh.geometry.union_all().centroid
    x_pas, y_pas = c_pas.x, c_pas.y
        
    # Ceibos / Chongón
    cei_hh = hh_gdf[hh_gdf['distrito_nom'].str.upper() == 'CEIBOS']
    c_cei = cei_hh.geometry.union_all().centroid
    x_cei, y_cei = c_cei.x, c_cei.y

    # 4. Crear la figura
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Ajustar márgenes para dar espacio al título superior
    plt.subplots_adjust(top=0.86, bottom=0.08, left=0.05, right=0.95)
    
    # Graficar por categoría para tener control total
    for cat, color in colors_dict.items():
        sub_gdf = gdf_urban[gdf_urban['lisa_category'] == cat]
        if not sub_gdf.empty:
            sub_gdf.plot(
                ax=ax,
                color=color,
                edgecolor='#ffffff', # Borde blanco ultra-fino para un estilo limpio
                linewidth=0.25,
                antialiased=True,
                zorder=2
            )
            
    # Graficar límites parroquiales de fondo para referencia si están disponibles
    gdf_parr = None
    if os.path.exists(config.SHAPEFILE_PATH):
        try:
            gdf_shp = gpd.read_file(config.SHAPEFILE_PATH)
            gdf_gye_shp = gdf_shp[gdf_shp['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
            if gdf_gye_shp.crs is None:
                gdf_gye_shp = gdf_gye_shp.set_crs("EPSG:32717")
            gdf_parr = gdf_gye_shp.dissolve(by='DPA_PARROQ', as_index=False).to_crs("EPSG:32717")
            gdf_parr_filtered = gdf_parr[gdf_parr['DPA_DESPAR'].str.upper() != 'PUNA'].copy()
            gdf_parr_filtered.plot(ax=ax, facecolor='none', edgecolor='#babcbe', linewidth=0.6, linestyle='--', alpha=0.5, zorder=3)
        except Exception as e:
            print(f"Advertencia: No se pudo superponer los límites parroquiales: {e}")
            
    # Ajustar encuadre espacial para centrarse perfectamente en la zona urbana
    xmin, ymin, xmax, ymax = gdf_urban.total_bounds
    x_pad = (xmax - xmin) * 0.03
    y_pad = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    ax.axis('off')
    
    # 5. ANOTACIONES CON LÍNEAS GUÍA (CALLOUTS)
    # A. Nueva Prosperina (Hotspot Norte-Oeste)
    ax.annotate(
        "NUEVA PROSPERINA\nClúster Alto-Alto (Crítico)",
        xy=(x_nva, y_nva),
        xytext=(x_nva - 13000, y_nva + 2000),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9.5, fontweight='bold', color='#111827',
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#e5e7eb", lw=1, alpha=0.94),
        zorder=10
    )
    
    # B. Sur / Guasmo (Hotspot Sur)
    ax.annotate(
        "SUR (GUASMO)\nClúster Alto-Alto (Crítico)",
        xy=(x_sur, y_sur),
        xytext=(x_sur + 6000, y_sur - 2500),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9.5, fontweight='bold', color='#111827',
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#e5e7eb", lw=1, alpha=0.94),
        zorder=10
    )
    
    # C. Pascuales (Hotspot Norte)
    ax.annotate(
        "PASCUALES\nClúster Alto-Alto",
        xy=(x_pas, y_pas),
        xytext=(x_pas + 5000, y_pas + 3000),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9.5, fontweight='bold', color='#111827',
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#e5e7eb", lw=1, alpha=0.94),
        zorder=10
    )
    
    # D. Ceibos / Chongón (Hotspot Oeste)
    ax.annotate(
        "CEIBOS / CHONGÓN\nClúster Alto-Alto",
        xy=(x_cei, y_cei),
        xytext=(x_cei - 12000, y_cei - 3000),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9.5, fontweight='bold', color='#111827',
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#e5e7eb", lw=1, alpha=0.94),
        zorder=10
    )
    
    # 6. TÍTULO Y LEYENDA
    ax.text(
        0.02, 0.98, "IDENTIFICANDO LOS BARRIOS MÁS PELIGROSOS",
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        color='#111827',
        ha='left',
        va='top'
    )
    
    ax.text(
        0.02, 0.94, 
        "Mapa de Clústeres de Autocorrelación Espacial Local (LISA)\nIdentificación de Hotspots significativos donde la violencia extrema está espacialmente correlacionada.",
        transform=ax.transAxes,
        fontsize=10.5,
        color='#4b5563',
        ha='left',
        va='top',
        linespacing=1.35
    )
    
    # Leyenda personalizada con diseño premium
    patches_list = [
        mpatches.Patch(color=color, label=legend_labels[cat]) 
        for cat, color in colors_dict.items()
    ]
    ax.legend(
        handles=patches_list, 
        loc='lower left', 
        fontsize=9.5, 
        title="Categorías LISA",
        title_fontsize=10.5,
        frameon=True,
        facecolor='white',
        edgecolor='#e5e7eb',
        framealpha=0.95
    )
    
    # Dibujar escala métrica discreta de 5 km
    scale_len = 5000  # 5 km
    scale_x = xmin + (xmax - xmin) * 0.05
    scale_y = ymin + (ymax - ymin) * 0.03
    tick_h = (ymax - ymin) * 0.007
    
    ax.plot([scale_x, scale_x + scale_len], [scale_y, scale_y], color='#374151', linewidth=2, zorder=5)
    ax.plot([scale_x, scale_x], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax.plot([scale_x + scale_len, scale_x + scale_len], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax.text(scale_x + scale_len/2, scale_y - tick_h * 1.8, '5 km', ha='center', va='top', fontsize=9.5, fontweight='semibold', color='#374151', zorder=5)
    
    # Guardar mapa
    out_name = "slide20_puntos_calientes_lisa"
    out_png_path = os.path.join(config.FIGURES_DIR, f"{out_name}.png")
    out_pdf_path = os.path.join(config.FIGURES_DIR, f"{out_name}.pdf")
    
    plt.savefig(out_png_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.savefig(out_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    
    print(f"Guardado local: {out_png_path}")
    
    # Copiar a carpeta de artefactos de Gemini
    if os.path.exists(ARTIFACT_DIR):
        dest_path = os.path.join(ARTIFACT_DIR, f"{out_name}.png")
        shutil.copy(out_png_path, dest_path)
        print(f"Copiado a artefactos: {dest_path}")

if __name__ == "__main__":
    generate_slide20_lisa_map()
