import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar el directorio actual al path para importar configuración
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config

def main():
    print("=" * 60)
    # 1. Cargar datos agregados anuales
    anual_csv_path = os.path.join(config.CLEAN_DATA_DIR, "datos_agregados_anual.csv")
    if not os.path.exists(anual_csv_path):
        print(f"Error: No se encontró el archivo de datos anuales en: {anual_csv_path}")
        print("Por favor ejecuta primero 'run_pipeline.py' para procesar los datos.")
        return
        
    print(f"Cargando datos agregados desde: {anual_csv_path}")
    df_anual = pd.read_csv(anual_csv_path)
    
    # Agrupar por subcircuito para obtener homicidios acumulados totales
    # Esto coincide exactamente con el cálculo de la Etapa 6 de tu pipeline
    df_sub = df_anual.groupby('cod_subcircu').agg({
        'homicidios': 'sum',
        'Area_km2': 'first'
    }).reset_index()
    
    y = df_sub['homicidios'].values
    
    # 2. Calcular los tres estadísticos clave
    mean_val = np.mean(y)
    var_val = np.var(y, ddof=1)
    ratio_val = var_val / mean_val
    
    # 3. Generar el histograma (Figura 3.9)
    print("\nGenerando Figura 3.9: Histograma de Homicidios...")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Dibujar histograma con curva de densidad ajustada (KDE)
    sns.histplot(
        y, kde=True, bins=25, ax=ax,
        color='#1f77b4', edgecolor='black', linewidth=1.0, alpha=0.6,
        line_kws={'linewidth': 2.5, 'color': '#d7191c', 'label': 'Curva de Densidad (KDE)'}
    )
    
    # Añadir línea vertical para la Media
    plt.axvline(
        mean_val, color='black', linestyle='--', linewidth=1.8,
        label=f'Media del conteo = {mean_val:.2f}'
    )
    
    # Personalización del gráfico
    plt.xlabel("Número de Homicidios Acumulados por Unidad Espacial", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("Frecuencia (Número de Unidades Espaciales)", fontsize=11, fontweight='bold', labelpad=10)
    plt.title("Figura 3.9: Distribución de Frecuencias de Homicidios Totales por Unidad Espacial\n(Cantón Guayaquil, 2024-2026)", 
              fontsize=12, fontweight='bold', pad=15)
    
    # Añadir anotación sobre la asimetría y el exceso de ceros/valores bajos
    text_info = (
        f"Asimetría Positiva Extrema\n"
        f"Media: {mean_val:.2f}\n"
        f"Varianza: {var_val:.2f}\n"
        f"Ratio Varianza/Media: {ratio_val:.2f} (Sobredispersión)"
    )
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#cccccc", lw=0.8)
    ax.text(0.95, 0.70, text_info, transform=ax.transAxes, ha='right', va='top', 
            fontsize=10, fontweight='bold', color='#333333', bbox=bbox_props)
    
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    
    # Guardar gráficos
    png_path = os.path.join(config.FIGURES_DIR, "Figura3.9_histograma_homicidios.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "Figura3.9_histograma_homicidios.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, format='pdf')
    plt.close()
    
    # 4. Imprimir salida formateada en consola
    print("\n" + "=" * 60)
    print(" TABLA DE SOBREDISPERSIÓN (PARA LA TESIS) ")
    print("=" * 60)
    print("| Indicador | Valor |")
    print("|---|---|")
    print(f"| Media de homicidios | {mean_val:.4f} |")
    print(f"| Varianza | {var_val:.4f} |")
    print(f"| Relación Varianza / Media | {ratio_val:.4f} |")
    print("=" * 60)
    print(f"\n[OK] Gráficos guardados en:\n- {png_path}\n- {pdf_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
