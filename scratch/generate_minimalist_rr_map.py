import os
import sys
import shutil
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.patches as patches

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Artifact directory for embedding images
ARTIFACT_DIR = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\8c0ff45f-78e7-412d-8e2e-8062467c5b75"

def generate_maps():
    print("Cargando datos...")
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
    if not os.path.exists(geojson_path):
        geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
        
    if not os.path.exists(geojson_path):
        print(f"Error: No se encontró el archivo GeoJSON en {geojson_path}")
        return
        
    gdf = gpd.read_file(geojson_path)
    
    # Check if rr_estimado_mediana column exists
    if 'rr_estimado_mediana' not in gdf.columns:
        csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
        if os.path.exists(csv_path):
            df_rr = pd.read_csv(csv_path)
            gdf = gdf.merge(df_rr[['cod_subcircu', 'rr_estimado_mediana', 'probabilidad_excedencia']], on='cod_subcircu', how='left')
        else:
            print("Error: No se encontraron los resultados de riesgo relativo.")
            return

    # Proyectar a UTM 17S (EPSG:32717)
    if gdf.crs != "EPSG:32717":
        gdf = gdf.to_crs("EPSG:32717")

    # Filtrar zonas rurales alejadas para zoom urbano limpio
    # Excluimos Puná (en distrito SUR), Tenguel (en distrito MODELO) y el distrito PROGRESO completo
    gdf_urban = gdf[
        (~gdf['subcircuito_nom'].str.upper().str.contains('PUNA|TENGUEL|MORRO|POSORJA|PROGRESO', na=False)) &
        (~gdf['distrito_nom'].str.upper().str.contains('PROGRESO|PLAYAS|SAMBORONDON|DAULE|NARANJAL', na=False))
    ].copy().dropna(subset=['rr_estimado_mediana'])

    print(f"Subcircuitos urbanos filtrados: {len(gdf_urban)}")
    
    # Valores de RR urbanos
    rr_vals = gdf_urban['rr_estimado_mediana'].values
    median_val = np.median(rr_vals)
    mean_val = np.mean(rr_vals)
    min_val = rr_vals.min()
    max_val = rr_vals.max()
    
    # -------------------------------------------------------------
    # 1. MAPA SECUENCIAL CONTINUO (SAND TO CRIMSON)
    # -------------------------------------------------------------
    print("Generando Mapa Secuencial Continuo...")
    # Gradiente sofisticado: crema cálido -> coral -> carmesí oscuro
    seq_colors = ['#fdfbf7', '#fca5a5', '#dc2626', '#7f1d1d']
    seq_cmap = LinearSegmentedColormap.from_list('minimalist_seq', seq_colors, N=256)
    
    # Usamos percentiles para normalizar y evitar que outliers de riesgo muy altos arruinen el contraste
    # Truncamos visualmente el máximo en el percentil 95
    p95 = np.percentile(rr_vals, 95)
    norm_seq = plt.Normalize(vmin=min_val, vmax=p95)
    
    fig, ax = plt.subplots(figsize=(10, 11), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    gdf_urban.plot(
        column='rr_estimado_mediana',
        cmap=seq_cmap,
        norm=norm_seq,
        ax=ax,
        edgecolor='#ffffff',
        linewidth=0.2,
        antialiased=True
    )
    ax.axis('off')
    
    # Título minimalista
    ax.text(0.02, 0.98, "Riesgo Relativo de Homicidios", transform=ax.transAxes,
            fontsize=18, fontweight='bold', color='#111827', ha='left', va='top')
    ax.text(0.02, 0.94, f"Modelo Espacial Bayesiano ICAR (Detalle Urbano)\nEscala Continua Secuencial (Percentil 95: {p95:.2f}x)", transform=ax.transAxes,
            fontsize=10, color='#4b5563', ha='left', va='top', linespacing=1.3)
    
    # Colorbar
    cax = fig.add_axes([0.15, 0.08, 0.70, 0.02])
    sm = plt.cm.ScalarMappable(cmap=seq_cmap, norm=norm_seq)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', drawedges=False)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, colors='#4b5563', length=0, pad=8)
    cbar.set_label('Intensidad de Riesgo Relativo', color='#374151', fontsize=10, fontweight='semibold', labelpad=-45)
    
    save_and_copy(fig, "mapa_rr_minimalista_secuencial_urbano")
    
    # -------------------------------------------------------------
    # 2. MAPA DIVERGENTE (CENTERED ON URBAN MEDIAN)
    # -------------------------------------------------------------
    print("Generando Mapa Divergente (Centrado en la Mediana local)...")
    # Azul elegante -> Blanco/Crema -> Rojo carmesí
    div_colors = [
        (0.0, '#1e3a8a'),  # Azul marino oscuro
        (0.3, '#3b82f6'),  # Azul medio
        (0.5, '#fcfcfc'),  # Blanco puro
        (0.7, '#ef4444'),  # Rojo
        (1.0, '#7f1d1d')   # Rojo oscuro
    ]
    div_cmap = LinearSegmentedColormap.from_list('minimalist_div', div_colors, N=256)
    
    # Centramos en la mediana local para ver zonas arriba/abajo de la mediana
    # Truncamos los extremos al p5 y p95 para mejorar visibilidad
    p5 = np.percentile(rr_vals, 5)
    norm_div = TwoSlopeNorm(vmin=p5, vcenter=median_val, vmax=p95)
    
    fig, ax = plt.subplots(figsize=(10, 11), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    gdf_urban.plot(
        column='rr_estimado_mediana',
        cmap=div_cmap,
        norm=norm_div,
        ax=ax,
        edgecolor='#ffffff',
        linewidth=0.2,
        antialiased=True
    )
    ax.axis('off')
    
    ax.text(0.02, 0.98, "Riesgo Relativo de Homicidios", transform=ax.transAxes,
            fontsize=18, fontweight='bold', color='#111827', ha='left', va='top')
    ax.text(0.02, 0.94, f"Modelo Espacial Bayesiano ICAR (Detalle Urbano)\nEscala Divergente (Azul: Menor Riesgo | Rojo: Mayor Riesgo | Mediana: {median_val:.2f}x)", transform=ax.transAxes,
            fontsize=10, color='#4b5563', ha='left', va='top', linespacing=1.3)
    
    # Colorbar
    cax = fig.add_axes([0.15, 0.08, 0.70, 0.02])
    sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=norm_div)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', drawedges=False)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=9, colors='#4b5563', length=0, pad=8)
    cbar.set_label('Desviación respecto a la Mediana Urbana', color='#374151', fontsize=10, fontweight='semibold', labelpad=-45)
    
    save_and_copy(fig, "mapa_rr_minimalista_divergente_urbano")

    # -------------------------------------------------------------
    # 3. MAPA CLASIFICADO EN QUINTILES LOCALES (DISCRETO)
    # -------------------------------------------------------------
    print("Generando Mapa de Quintiles...")
    # Calcular quintiles de la zona urbana
    q = np.quantile(rr_vals, [0.2, 0.4, 0.6, 0.8])
    labels = [
        f'Muy Bajo (RR ≤ {q[0]:.2f})',
        f'Bajo ({q[0]:.2f} - {q[1]:.2f})',
        f'Medio ({q[1]:.2f} - {q[2]:.2f})',
        f'Alto ({q[2]:.2f} - {q[3]:.2f})',
        f'Muy Alto (RR > {q[3]:.2f})'
    ]
    gdf_urban['rr_quintil_local'] = pd.qcut(gdf_urban['rr_estimado_mediana'], q=5, labels=labels)
    
    # Colores discretos elegantes (ColorBrewer RdYlBu invertido o similar modificado)
    # Azul suave -> Azul claro -> Blanco/Crema -> Naranja -> Rojo coral
    discrete_colors = ['#2c7bb6', '#abd9e9', '#fefefe', '#fdae61', '#d7191c']
    disc_cmap = LinearSegmentedColormap.from_list('minimalist_disc', discrete_colors, N=5)
    
    fig, ax = plt.subplots(figsize=(10, 11), facecolor='#ffffff')
    ax.set_facecolor('#ffffff')
    
    gdf_urban.plot(
        column='rr_quintil_local',
        cmap=disc_cmap,
        legend=True,
        legend_kwds={
            'title': 'Nivel de Riesgo (Quintiles Locales)',
            'loc': 'lower left',
            'fontsize': 9,
            'title_fontsize': 10,
            'frameon': False,  # Sin recuadro para estilo minimalista
        },
        ax=ax,
        edgecolor='#ffffff',
        linewidth=0.2,
        antialiased=True
    )
    ax.axis('off')
    
    ax.text(0.02, 0.98, "Riesgo Relativo de Homicidios", transform=ax.transAxes,
            fontsize=18, fontweight='bold', color='#111827', ha='left', va='top')
    ax.text(0.02, 0.94, "Modelo Espacial Bayesiano ICAR (Detalle Urbano)\nClasificación en Quintiles Locales (20% de subcircuitos por clase)", transform=ax.transAxes,
            fontsize=10, color='#4b5563', ha='left', va='top', linespacing=1.3)
    
    save_and_copy(fig, "mapa_rr_minimalista_quintiles_urbano")

def save_and_copy(fig, name):
    # Guardar en outputs/figures
    out_png_path = os.path.join(config.FIGURES_DIR, f"{name}.png")
    out_pdf_path = os.path.join(config.FIGURES_DIR, f"{name}.pdf")
    fig.savefig(out_png_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    fig.savefig(out_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    print(f"Guardado local: {out_png_path}")
    
    # Copiar a carpeta de artefactos de Gemini
    if os.path.exists(ARTIFACT_DIR):
        dest_path = os.path.join(ARTIFACT_DIR, f"{name}.png")
        shutil.copy(out_png_path, dest_path)
        print(f"Copiado a artefactos: {dest_path}")

if __name__ == "__main__":
    generate_maps()

