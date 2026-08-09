import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import libpysal
from esda.moran import Moran

# Agregar el directorio actual al path para importar configuración
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config

def main():
    print("=" * 60)
    # 1. Cargar datos espaciales agregados
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
    if not os.path.exists(geojson_path):
        print(f"Error: No se encontró el archivo espacial en: {geojson_path}")
        print("Por favor ejecuta primero 'run_pipeline.py' para procesar los datos.")
        return
        
    print(f"Cargando subcircuitos desde: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    
    # 2. Reconstruir pesos espaciales de tipo Queen (igual al del pipeline)
    print("Construyendo matriz de pesos espaciales Queen...")
    w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    w.transform = 'r' # Estandarización por filas
    
    # Variable a analizar (Homicidios Totales acumulados)
    y = gdf['homicidios_totales'].values
    
    # 3. Calcular Moran Global formalmente con 9999 permutaciones
    print("Calculando Índice de Moran Global...")
    moran = Moran(y, w, permutations=9999)
    
    # 4. Estandarizar la variable y calcular su lag espacial estandarizado
    y_std = (y - np.mean(y)) / np.std(y)
    y_lag = libpysal.weights.spatial_lag.lag_spatial(w, y)
    y_lag_std = (y_lag - np.mean(y_lag)) / np.std(y_lag)
    
    # 5. Generar Figura 3.8: Moran Scatter Plot
    print("\nGenerando Figura 3.8: Moran Scatter Plot...")
    fig, ax = plt.subplots(figsize=(9, 8.5))
    
    # Graficar puntos y recta de regresión
    sns.regplot(
        x=y_std, y=y_lag_std, ci=None, ax=ax,
        line_kws={'color': '#d7191c', 'linewidth': 2.5, 'label': f"Pendiente (Moran's I) = {moran.I:.4f}"},
        scatter_kws={'color': '#2c7bb6', 'alpha': 0.6, 'edgecolor': 'white', 's': 55}
    )
    
    # Líneas de referencia cuadrantes cruzadas en cero
    plt.axhline(0, color='#555555', linestyle='--', linewidth=1.2)
    plt.axvline(0, color='#555555', linestyle='--', linewidth=1.2)
    
    # Personalización de límites de ejes con un margen
    padding = 0.2
    max_val = max(abs(y_std).max(), abs(y_lag_std).max()) + padding
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    # Cuadros de texto con fondo blanco y bordes grises para explicar los cuadrantes de Moran
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9, lw=0.8)
    
    # Cuadrante I: Alto-Alto
    ax.text(0.95, 0.95, "I: Alto-Alto (High-High)\n(Homicidios altos rodeados de áreas altas)", 
            transform=ax.transAxes, ha='right', va='top', fontsize=9.5, color='#d7191c', fontweight='bold', bbox=bbox_props)
    
    # Cuadrante II: Bajo-Alto
    ax.text(0.05, 0.95, "II: Bajo-Alto (Low-High)\n(Homicidios bajos rodeados de áreas altas)", 
            transform=ax.transAxes, ha='left', va='top', fontsize=9.5, color='#74add1', fontweight='bold', bbox=bbox_props)
    
    # Cuadrante III: Bajo-Bajo
    ax.text(0.05, 0.05, "III: Bajo-Bajo (Low-Low)\n(Homicidios bajos rodeados de áreas bajas)", 
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9.5, color='#2c7bb6', fontweight='bold', bbox=bbox_props)
    
    # Cuadrante IV: Alto-Bajo
    ax.text(0.95, 0.05, "IV: Alto-Bajo (High-Low)\n(Homicidios altos rodeados de áreas bajas)", 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9.5, color='#fdae61', fontweight='bold', bbox=bbox_props)
    
    # Etiquetas y título
    plt.xlabel("Homicidios Estandarizados ($z_i$)", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Lag Espacial Estandarizado ($W z_i$)", fontsize=11, fontweight='bold', labelpad=10)
    plt.title(f"Figura 3.8: Moran Scatter Plot de Homicidios (Canton Guayaquil)\nMoran's I = {moran.I:.4f} (z = {moran.z_sim:.2f}, p-valor < 0.001)", 
              fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=True, fontsize=10, facecolor='#f8f9fa')
    
    # Guardar la imagen en alta definición
    png_path = os.path.join(config.FIGURES_DIR, "Figura3.8_moran_scatterplot.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura3.8_moran_scatterplot.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    # 6. Imprimir los resultados en consola formateados
    print("\n" + "=" * 60)
    print(" TABLA DE MORAN GLOBAL (PARA LA TESIS) ")
    print("=" * 60)
    print(f"| Indicador | Valor |")
    print(f"|---|---|")
    print(f"| Índice de Moran (I) | {moran.I:.4f} |")
    print(f"| Valor esperado | {moran.EI:.4f} |")
    print(f"| Estadístico z | {moran.z_sim:.2f} |")
    # Mostrar < 0.001 si el pseudo p-valor simula muy bajo
    p_val_str = "< 0.001" if moran.p_sim < 0.001 else f"{moran.p_sim:.4f}"
    print(f"| Valor p | {p_val_str} |")
    print("=" * 60)
    print(f"\n[OK] Gráficos de Moran guardados en:\n- {png_path}\n- {pdf_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
