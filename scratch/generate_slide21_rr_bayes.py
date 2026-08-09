import os
import sys
import shutil
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Artifact directory for embedding images
ARTIFACT_DIR = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\8c0ff45f-78e7-412d-8e2e-8062467c5b75"

def generate_slide21_graphic():
    print("Cargando datos geográficos...")
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
    if not os.path.exists(geojson_path):
        geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
        
    if not os.path.exists(geojson_path):
        print(f"Error: No se encontró el GeoJSON en {geojson_path}")
        return
        
    gdf = gpd.read_file(geojson_path)
    
    # Check if rr_estimado_mediana exists
    if 'rr_estimado_mediana' not in gdf.columns:
        csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
        if os.path.exists(csv_path):
            df_rr = pd.read_csv(csv_path)
            gdf = gdf.merge(df_rr[['cod_subcircu', 'rr_estimado_mediana', 'rr_limite_inf_95', 'rr_limite_sup_95', 'probabilidad_excedencia']], on='cod_subcircu', how='left')
        else:
            print("Error: No se encontraron los resultados de riesgo relativo.")
            return

    # Proyectar a UTM 17S (EPSG:32717)
    if gdf.crs != "EPSG:32717":
        gdf = gdf.to_crs("EPSG:32717")

    # 1. Filtrar el distrito ESTEROS para el panel A
    gdf_esteros = gdf[gdf['distrito_nom'].str.upper() == 'ESTEROS'].copy().dropna(subset=['rr_estimado_mediana'])
    print(f"Subcircuitos en distrito Esteros: {len(gdf_esteros)}")
    
    # 2. Configurar figura de doble panel (16:9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9.5), facecolor='#ffffff')
    fig.patch.set_facecolor('#ffffff')
    plt.subplots_adjust(top=0.84, bottom=0.08, left=0.06, right=0.94, wspace=0.25)
    
    # -------------------------------------------------------------
    # PANEL 1: MAPA DETALLADO DEL DISTRITO ESTEROS
    # -------------------------------------------------------------
    print("Graficando Panel A: Mapa de Esteros...")
    
    # Colormap secuencial cálido elegante (crema -> coral -> rojo oscuro)
    seq_colors = ['#fdfbf7', '#fca5a5', '#dc2626', '#7f1d1d']
    seq_cmap = LinearSegmentedColormap.from_list('esteros_seq', seq_colors, N=256)
    
    # Normalizar usando los límites locales de Esteros
    vmin_est = gdf_esteros['rr_estimado_mediana'].min()
    vmax_est = gdf_esteros['rr_estimado_mediana'].max()
    norm_est = plt.Normalize(vmin=vmin_est, vmax=vmax_est)
    
    # Dibujar subcircuitos
    gdf_esteros.plot(
        column='rr_estimado_mediana',
        cmap=seq_cmap,
        norm=norm_est,
        ax=ax1,
        edgecolor='#babcbe',
        linewidth=0.4,
        antialiased=True,
        zorder=2
    )
    
    # Destacar específicamente TRINITARIA NORTE 4
    trin_4 = gdf_esteros[gdf_esteros['subcircuito_nom'] == 'TRINITARIA NORTE 4']
    if not trin_4.empty:
        trin_4.plot(
            ax=ax1,
            facecolor='none',
            edgecolor='#111827',  # Borde oscuro grueso
            linewidth=2.0,
            zorder=4
        )
        # Calcular centroide para anotación
        c_trin = trin_4.geometry.centroid.values[0]
        x_trin, y_trin = c_trin.x, c_trin.y
        
        # Anotar Trinitaria Norte 4
        ax1.annotate(
            "TRINITARIA NORTE 4\nRR Mediana = 23.97x\nProb(RR > 1.5) = 100.0%",
            xy=(x_trin, y_trin),
            xytext=(x_trin - 3200, y_trin + 2000),
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle="arc3,rad=-0.1",
                fc="#1f2937", ec="#374151", lw=1.2,
                shrinkA=0, shrinkB=6
            ),
            fontsize=9.5, fontweight='bold', color='#111827',
            bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#111827", lw=1.2, alpha=0.95),
            zorder=10
        )
        
    ax1.axis('off')
    ax1.set_title("A. Zoom Espacial: Distrito Esteros (Focos de Riesgo)", fontsize=13, fontweight='semibold', color='#374151', pad=12)
    
    # Barra de escala métrica para ax1 (2 km)
    scale_len = 2000
    xmin, ymin, xmax, ymax = gdf_esteros.total_bounds
    scale_x = xmin + (xmax - xmin) * 0.05
    scale_y = ymin + (ymax - ymin) * 0.04
    tick_h = (ymax - ymin) * 0.008
    
    ax1.plot([scale_x, scale_x + scale_len], [scale_y, scale_y], color='#374151', linewidth=2, zorder=5)
    ax1.plot([scale_x, scale_x], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax1.plot([scale_x + scale_len, scale_x + scale_len], [scale_y - tick_h/2, scale_y + tick_h/2], color='#374151', linewidth=1, zorder=5)
    ax1.text(scale_x + scale_len/2, scale_y - tick_h * 1.8, '2 km', ha='center', va='top', fontsize=9, fontweight='semibold', color='#374151', zorder=5)

    # Colorbar para el mapa
    cax = fig.add_axes([0.06, 0.08, 0.40, 0.015])
    sm = plt.cm.ScalarMappable(cmap=seq_cmap, norm=norm_est)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', drawedges=False)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8.5, colors='#4b5563', length=0, pad=6)
    cbar.set_label('Riesgo Relativo Mediano Estimado (Escala Local)', color='#374151', fontsize=9.5, fontweight='semibold', labelpad=-40)

    # -------------------------------------------------------------
    # PANEL 2: GRÁFICO DE BARRAS DE TOP 10 SUBCIRCUITOS
    # -------------------------------------------------------------
    print("Graficando Panel B: Top 10 Subcircuitos...")
    
    # Extraer los 10 subcircuitos con mayor riesgo relativo de todo Guayaquil
    df_all = pd.DataFrame(gdf.drop(columns='geometry'))
    top_10 = df_all.sort_values(by='rr_estimado_mediana', ascending=False).head(10).copy()
    
    # Invertir el orden para que el primero quede arriba
    top_10 = top_10.iloc[::-1]
    
    # Definir colores de las barras: Rojo para Trinitaria Norte 4, Gris para el resto
    colors = ['#dc2626' if name == 'TRINITARIA NORTE 4' else '#9ca3af' for name in top_10['subcircuito_nom']]
    
    # Dibujar barras horizontales
    y_pos = np.arange(len(top_10))
    bars = ax2.barh(y_pos, top_10['rr_estimado_mediana'], color=colors, height=0.6, edgecolor='none')
    
    # Personalizar ejes
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_10['subcircuito_nom'], fontsize=9.5, fontweight='bold', color='#374151')
    ax2.set_xlabel('Riesgo Relativo Estimado (Mediana)', fontsize=10.5, color='#374151', labelpad=8)
    ax2.set_xlim(0, 30)  # Dejar espacio para las etiquetas a la derecha de las barras
    
    # Quitar bordes para limpieza minimalista
    for spine in ['top', 'right', 'left']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_color('#cccccc')
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3, color='#cccccc')
    ax2.set_axisbelow(True)
    
    # Agregar etiquetas con los valores y los intervalos de credibilidad
    for bar, idx in zip(bars, top_10.index):
        row = top_10.loc[idx]
        val = row['rr_estimado_mediana']
        inf = row['rr_limite_inf_95']
        sup = row['rr_limite_sup_95']
        prob = row['probabilidad_excedencia'] * 100.0
        
        # Etiqueta de texto: Mediana [IC 95%]
        label_text = f" {val:.2f}x  [IC 95%: {inf:.1f} - {sup:.1f}]  (P={prob:.1f}%)"
        
        # Determinar color de texto: rojo si es Trinitaria, gris oscuro para el resto
        txt_color = '#b91c1c' if row['subcircuito_nom'] == 'TRINITARIA NORTE 4' else '#1f2937'
        weight = 'bold' if row['subcircuito_nom'] == 'TRINITARIA NORTE 4' else 'normal'
        
        ax2.text(
            val, bar.get_y() + bar.get_height()/2, label_text,
            ha='left', va='center', fontsize=8.5, color=txt_color, fontweight=weight
        )
        
    ax2.set_title("B. Top 10 Subcircuitos con Mayor Riesgo Relativo de Guayaquil", fontsize=13, fontweight='semibold', color='#374151', pad=12)

    # -------------------------------------------------------------
    # DETALLES ESTÉTICOS GLOBALES
    # -------------------------------------------------------------
    plt.suptitle("EL RIESGO RELATIVO BAYESIANO Y CERTEZA PROBABILÍSTICA", fontsize=20, fontweight='bold', color='#111827', y=0.97)
    fig.text(
        0.5, 0.91, 
        "Comparación espacial del riesgo en relación al promedio cantonal (RR = 1.0) e intervalos de credibilidad del 95% posterior.",
        fontsize=11.5, color='#4b5563', ha='center', style='italic'
    )
    
    # Guardar
    out_name = "slide21_riesgo_relativo_bayesiano"
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
    generate_slide21_graphic()
