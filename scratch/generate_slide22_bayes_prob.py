import os
import sys
import shutil
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Artifact directory for embedding images
ARTIFACT_DIR = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\8c0ff45f-78e7-412d-8e2e-8062467c5b75"

def generate_slide22_map():
    print("Cargando datos geográficos...")
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
    if not os.path.exists(geojson_path):
        geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
        
    if not os.path.exists(geojson_path):
        print(f"Error: No se encontró el GeoJSON en {geojson_path}")
        return
        
    gdf = gpd.read_file(geojson_path)
    
    # Check if exceedance probability exists
    if 'probabilidad_excedencia' not in gdf.columns:
        csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
        if os.path.exists(csv_path):
            df_rr = pd.read_csv(csv_path)
            gdf = gdf.merge(df_rr[['cod_subcircu', 'probabilidad_excedencia']], on='cod_subcircu', how='left')
        else:
            print("Error: No se encontraron los resultados de riesgo relativo.")
            return

    # Proyectar a UTM 17S (EPSG:32717)
    if gdf.crs != "EPSG:32717":
        gdf = gdf.to_crs("EPSG:32717")

    # 1. Filtrar áreas urbanas (excluyendo distritos rurales distantes)
    gdf_urban = gdf[
        (~gdf['subcircuito_nom'].str.upper().str.contains('PUNA|TENGUEL|MORRO|POSORJA|PROGRESO', na=False)) &
        (~gdf['distrito_nom'].str.upper().str.contains('PROGRESO|PLAYAS|SAMBORONDON|DAULE|NARANJAL', na=False))
    ].copy()
      # 2. Categorizar la probabilidad posterior de excedencia (Certeza de sobre-riesgo)
    # Rojo: P > 90% (Certeza Alta)
    # Gris Claro: P 50% - 90% (Certeza Moderada)
    # Azul: P < 50% (Certeza Baja)
    def categorize_prob(p):
        if p > 0.90:
            return 'Certeza Alta de Sobre-Riesgo (P > 90%)'
        elif p >= 0.50:
            return 'Certeza Moderada/Indeterminada (P 50% - 90%)'
        else:
            return 'Baja Certeza de Sobre-Riesgo (P < 50%)'
            
    gdf_urban['prob_cat'] = gdf_urban['probabilidad_excedencia'].apply(categorize_prob)
    
    # Colores correspondientes
    color_map = {
        'Certeza Alta de Sobre-Riesgo (P > 90%)': '#ef4444',      # Rojo Coral
        'Certeza Moderada/Indeterminada (P 50% - 90%)': '#e5e7eb',  # Gris Claro Neutro
        'Baja Certeza de Sobre-Riesgo (P < 50%)': '#3b82f6'           # Azul Muted
    }
 
    # 3. Calcular centroides específicos para apuntar las anotaciones
    # Foco 1: Nueva Prosperina (Probabilidad 100%)
    np_gdf = gdf_urban[(gdf_urban['distrito_nom'].str.upper() == 'NUEVA PROSPERINA') & (gdf_urban['probabilidad_excedencia'] > 0.99)]
    c_np = np_gdf.geometry.union_all().centroid
    x_np, y_np = c_np.x, c_np.y
    
    # Foco 2: Sur / Guasmo (Probabilidad 100%)
    sur_gdf = gdf_urban[(gdf_urban['distrito_nom'].str.upper() == 'SUR') & (gdf_urban['probabilidad_excedencia'] > 0.99)]
    c_sur = sur_gdf.geometry.union_all().centroid
    x_sur, y_sur = c_sur.x, c_sur.y
    
    # Foco 3: Zona de Riesgo Bajo (Azul) - Generalmente en Ceibos o Centro
    bajo_gdf = gdf_urban[gdf_urban['prob_cat'] == 'Baja Certeza de Sobre-Riesgo (P < 50%)']
    c_bajo = bajo_gdf.geometry.union_all().centroid
    x_bajo, y_bajo = c_bajo.x, c_bajo.y
 
    # 4. Crear la figura
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Ajustar márgenes para dar espacio al título superior
    plt.subplots_adjust(top=0.86, bottom=0.08, left=0.05, right=0.95)
    
    # Graficar por categoría para control total
    for cat, color in color_map.items():
        sub_gdf = gdf_urban[gdf_urban['prob_cat'] == cat]
        if not sub_gdf.empty:
            sub_gdf.plot(
                ax=ax,
                color=color,
                edgecolor='#ffffff',  # Bordes blancos limpios
                linewidth=0.25,
                antialiased=True,
                zorder=2
            )
            
    # Límites parroquiales para superposición de referencia
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
 
    # Ajustar encuadre espacial
    xmin, ymin, xmax, ymax = gdf_urban.total_bounds
    x_pad = (xmax - xmin) * 0.03
    y_pad = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    ax.axis('off')
    
    # 5. ANOTACIONES CON LÍNEAS GUÍA (CALLOUTS)
    # A. Nueva Prosperina (Riesgo Muy Alto)
    ax.annotate(
        "NUEVA PROSPERINA\nCerteza de Sobre-Riesgo = 100%\n(P(RR > 1.5) = 1.0)\nHotspot Confirmado",
        xy=(x_np, y_np),
        xytext=(x_np - 14000, y_np + 2000),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9.5, fontweight='bold', color='#dc2626',
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#ef4444", lw=1.2, alpha=0.95),
        zorder=10
    )
    
    # B. Sur / Guasmo (Riesgo Muy Alto)
    ax.annotate(
        "SUR (GUASMO)\nCerteza de Sobre-Riesgo = 100%\n(P(RR > 1.5) = 1.0)\nHotspot Confirmado",
        xy=(x_sur, y_sur),
        xytext=(x_sur + 6000, y_sur - 2500),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9.5, fontweight='bold', color='#dc2626',
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#ef4444", lw=1.2, alpha=0.95),
        zorder=10
    )
    
    # C. Zona de Riesgo Bajo (Azul)
    ax.annotate(
        "CEIBOS / CENTRO-NORTE\nBaja Certeza de Sobre-Riesgo\n(P(RR > 1.5) < 50%)\nZona de Bajo Riesgo",
        xy=(x_bajo, y_bajo),
        xytext=(x_bajo - 12000, y_bajo - 4000),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9.5, fontweight='bold', color='#2563eb',
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#3b82f6", lw=1.2, alpha=0.95),
        zorder=10
    )
 
    # 6. TÍTULO Y LEYENDA
    ax.text(
        0.02, 0.98, "MAPA DE CERTEZA DE SOBRE-RIESGO BAYESIANO",
        transform=ax.transAxes,
        fontsize=20,
        fontweight='bold',
        color='#111827',
        ha='left',
        va='top'
    )
    
    ax.text(
        0.02, 0.94, 
        "Clasificación espacial de subcircuitos de acuerdo a la probabilidad posterior de que el riesgo relativo supere 1.5: P(RR > 1.5 | Datos).\nNota: Mide la certeza estadística de que el área tiene sobre-riesgo (>50% sobre el promedio cantonal), NO la probabilidad de que ocurra un homicidio.",
        transform=ax.transAxes,
        fontsize=10.5,
        color='#4b5563',
        ha='left',
        va='top',
        linespacing=1.35
    )
    
    # Leyenda personalizada con diseño premium
    patches_list = [
        mpatches.Patch(color=color, label=cat) 
        for cat, color in color_map.items()
    ]
    ax.legend(
        handles=patches_list, 
        loc='lower left', 
        fontsize=9.5, 
        title="Certeza de Riesgo Excedente P(RR > 1.5)",
        title_fontsize=10.5,
        frameon=True,
        facecolor='white',
        edgecolor='#e5e7eb',
        framealpha=0.95
    )
    
    # Escala de 5 km
    scale_len = 5000
    scale_x = xmin + (xmax - xmin) * 0.05
    scale_y = ymin + (ymax - ymin) * 0.03
    tick_h = (ymax - ymin) * 0.007
    
    ax.plot([scale_x, scale_x + scale_len], [scale_y, scale_y], color='#374151', linewidth=2, zorder=5)
    ax.plot([scale_x, scale_x], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax.plot([scale_x + scale_len, scale_x + scale_len], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax.text(scale_x + scale_len/2, scale_y - tick_h * 1.8, '5 km', ha='center', va='top', fontsize=9.5, fontweight='semibold', color='#374151', zorder=5)
    
    # Guardar
    out_name = "slide22_mapa_probabilidad_riesgo"
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
    generate_slide22_map()
