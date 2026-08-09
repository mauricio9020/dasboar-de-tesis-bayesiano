import os
import sys
import shutil
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from shapely.geometry import Point
import matplotlib.patches as patches

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Artifact directory for embedding images
ARTIFACT_DIR = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\8c0ff45f-78e7-412d-8e2e-8062467c5b75"

def generate_slide18_graphic():
    print("Cargando datos limpios...")
    clean_csv_path = os.path.join(config.CLEAN_DATA_DIR, "datos_limpios.csv")
    if not os.path.exists(clean_csv_path):
        print(f"Error: No se encontró el archivo de datos limpios en: {clean_csv_path}")
        return
        
    df = pd.read_csv(clean_csv_path).dropna(subset=['latitud', 'longitud'])
    
    # 1. Filtrar áreas urbanas (excluyendo distritos rurales alejados)
    df_urban = df[
        (~df['Subcircuito'].str.upper().str.contains('PUNA|TENGUEL|MORRO|POSORJA|PROGRESO', na=False)) &
        (~df['Distrito'].str.upper().str.contains('PROGRESO|PLAYAS|SAMBORONDON|DAULE|NARANJAL', na=False))
    ].copy()
    
    print(f"Total eventos urbanos: {len(df_urban)}")
    
    # 2. Convertir a GeoDataFrame y proyectar a Web Mercator (EPSG:3857)
    geometry = [Point(xy) for xy in zip(df_urban['longitud'], df_urban['latitud'])]
    gdf_urban = gpd.GeoDataFrame(df_urban, geometry=geometry, crs="EPSG:4326")
    gdf_urban_merc = gdf_urban.to_crs(epsg=3857)
    
    # 3. Obtener límites de encuadre con un 4% de margen
    xmin, ymin, xmax, ymax = gdf_urban_merc.total_bounds
    x_pad = (xmax - xmin) * 0.04
    y_pad = (ymax - ymin) * 0.04
    xlim = (xmin - x_pad, xmax + x_pad)
    ylim = (ymin - y_pad, ymax + y_pad)
    
    # 4. Calcular ubicaciones promedio de Norte, Centro-Oeste y Sur para las anotaciones
    pts_norte = gdf_urban_merc[gdf_urban_merc['Distrito'].str.upper().isin(['NUEVA PROSPERINA', 'PASCUALES'])]
    pts_centro_oeste = gdf_urban_merc[gdf_urban_merc['Distrito'].str.upper().isin(['PORTETE', 'FLORIDA'])]
    # El distrito Sur y Esteros concentran el hotspot del sur
    pts_sur = gdf_urban_merc[gdf_urban_merc['Distrito'].str.upper().isin(['SUR', 'ESTEROS'])]
    
    x_norte, y_norte = pts_norte.geometry.x.mean(), pts_norte.geometry.y.mean()
    # Ajustar para apuntar al núcleo de Nueva Prosperina
    x_norte -= 800
    y_norte += 800
    
    # Usar solo PORTETE para el Centro-Oeste para apuntar al hotspot real
    pts_co = gdf_urban_merc[gdf_urban_merc['Distrito'].str.upper() == 'PORTETE']
    x_co, y_co = pts_co.geometry.x.mean(), pts_co.geometry.y.mean()
    x_co -= 500
    
    x_sur, y_sur = pts_sur.geometry.x.mean(), pts_sur.geometry.y.mean()
    x_sur += 500
    
    # 5. Configurar la figura: dos paneles uno al lado del otro
    # 16:9 ratio aproximado para diapositiva (16 x 9.5 pulgadas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9.5), facecolor='#ffffff')
    fig.patch.set_facecolor('#ffffff')
    
    # Ajustar márgenes para evitar solapamientos con títulos y leyendas
    plt.subplots_adjust(top=0.82, bottom=0.08, left=0.06, right=0.94, wspace=0.18)
    
    # Cargar límites parroquiales para superposición tenue
    gdf_parr = None
    if os.path.exists(config.SHAPEFILE_PATH):
        try:
            gdf_shp = gpd.read_file(config.SHAPEFILE_PATH)
            gdf_gye_shp = gdf_shp[gdf_shp['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
            if gdf_gye_shp.crs is None:
                gdf_gye_shp = gdf_gye_shp.set_crs("EPSG:32717")
            gdf_parr = gdf_gye_shp.dissolve(by='DPA_PARROQ', as_index=False).to_crs(epsg=3857)
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el shapefile: {e}")
            
    # -------------------------------------------------------------
    # PANEL 1: MAPA DE PUNTOS
    # -------------------------------------------------------------
    print("Graficando Panel A: Mapa de Puntos...")
    
    # Superponer límites parroquiales si existen
    if gdf_parr is not None:
        gdf_parr.plot(ax=ax1, facecolor='none', edgecolor='#babcbe', linewidth=0.5, linestyle='--', alpha=0.5, zorder=2)
        
    # Graficar los puntos individuales
    gdf_urban_merc.plot(
        ax=ax1,
        color='#ef4444',     # Rojo coral vibrante
        markersize=2.5,      # Puntos definidos
        alpha=0.3,           # Transparencia para ver el solapamiento natural
        edgecolor='none',
        zorder=3
    )
    
    # Añadir basemap
    cx.add_basemap(ax1, source=cx.providers.CartoDB.Positron, alpha=0.55, zorder=1)
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.axis('off')
    ax1.set_title("A. Distribución de Casos (Homicidios)", fontsize=14, fontweight='semibold', color='#374151', pad=12)
    
    # -------------------------------------------------------------
    # PANEL 2: MAPA DE DENSIDAD KDE
    # -------------------------------------------------------------
    print("Graficando Panel B: Mapa de Densidad (KDE)...")
    
    # Superponer límites parroquiales
    if gdf_parr is not None:
        gdf_parr.plot(ax=ax2, facecolor='none', edgecolor='#babcbe', linewidth=0.5, linestyle='--', alpha=0.5, zorder=2)
        
    # Graficar KDE con paleta YlOrRd (Amarillo -> Naranja -> Rojo)
    sns.kdeplot(
        x=gdf_urban_merc.geometry.x,
        y=gdf_urban_merc.geometry.y,
        ax=ax2,
        cmap='YlOrRd',
        fill=True,
        alpha=0.65,
        thresh=0.03,         # Cortar el halo exterior muy bajo
        levels=15,
        gridsize=250,
        bw_adjust=1.3,
        cbar=True,
        cbar_kws={
            'shrink': 0.45,
            'pad': 0.02,
            'orientation': 'vertical',
            'label': 'Densidad Relativa de Homicidios'
        },
        zorder=3
    )
    
    # Ajustar etiqueta de la barra de color
    cbar_ax = fig.axes[-1]
    cbar_ax.set_yticks([])
    cbar_ax.tick_params(labelsize=9, colors='#4b5563', length=0)
    cbar_ax.yaxis.label.set_size(10)
    cbar_ax.yaxis.label.set_color('#374151')
    cbar_ax.yaxis.label.set_weight('semibold')
    cbar_ax.text(1.2, 0.02, 'Mínima', ha='left', va='center', fontsize=9, transform=cbar_ax.transAxes, color='#4b5563')
    cbar_ax.text(1.2, 0.98, 'Máxima', ha='left', va='center', fontsize=9, transform=cbar_ax.transAxes, color='#4b5563')
    
    # Añadir basemap
    cx.add_basemap(ax2, source=cx.providers.CartoDB.Positron, alpha=0.55, zorder=1)
    
    # ANOTACIONES EN PANEL B: NORTE, CENTRO-OESTE Y SUR (Posiciones optimizadas y flechas cortas)
    # 1. Norte
    ax2.annotate(
        "NORTE\n(Nva. Prosperina / Pascuales)",
        xy=(x_norte, y_norte),
        xytext=(x_norte - 9500, y_norte + 3000),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9, fontweight='bold', color='#111827',
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#9ca3af", lw=0.8, alpha=0.92),
        zorder=10
    )
    
    # 2. Centro-Oeste
    ax2.annotate(
        "CENTRO-OESTE\n(Portete)",
        xy=(x_co, y_co),
        xytext=(x_co - 8500, y_co - 2500),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9, fontweight='bold', color='#111827',
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#9ca3af", lw=0.8, alpha=0.92),
        zorder=10
    )
    
    # 3. Sur
    ax2.annotate(
        "SUR\n(Sur / Esteros)",
        xy=(x_sur, y_sur),
        xytext=(x_sur + 4500, y_sur - 3000),
        arrowprops=dict(
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.1",
            fc="#1f2937", ec="#4b5563", lw=1.2,
            shrinkA=0, shrinkB=4
        ),
        fontsize=9, fontweight='bold', color='#111827',
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#9ca3af", lw=0.8, alpha=0.92),
        zorder=10
    )

    
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.axis('off')
    ax2.set_title("B. Densidad Kernel de Eventos (KDE)", fontsize=14, fontweight='semibold', color='#374151', pad=10)
    
    # -------------------------------------------------------------
    # DETALLES ESTÉTICOS GLOBALES
    # -------------------------------------------------------------
    # Título principal y mensaje clave de la diapositiva en la parte superior
    plt.suptitle("EL CRIMEN NO ES ALEATORIO", fontsize=22, fontweight='bold', color='#111827', y=0.97)
    fig.text(
        0.5, 0.91, 
        "Los mapas de eventos y calor demuestran una clara segregación espacial con concentración crítica en los ejes Norte, Centro-Oeste y Sur.",
        fontsize=12, color='#4b5563', ha='center', style='italic'
    )
    
    # Barra de escala discreta en el panel A
    scale_len = 5000  # 5 km
    scale_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    scale_y = ylim[0] + (ylim[1] - ylim[0]) * 0.04
    tick_h = (ylim[1] - ylim[0]) * 0.008
    
    ax1.plot([scale_x, scale_x + scale_len], [scale_y, scale_y], color='#374151', linewidth=2, zorder=5)
    ax1.plot([scale_x, scale_x], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax1.plot([scale_x + scale_len, scale_x + scale_len], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax1.text(scale_x + scale_len/2, scale_y - tick_h * 1.8, '5 km', ha='center', va='top', fontsize=9, fontweight='semibold', color='#374151', zorder=5)
    
    # Guardar y copiar
    out_name = "slide18_analisis_espacial"
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
    generate_slide18_graphic()
