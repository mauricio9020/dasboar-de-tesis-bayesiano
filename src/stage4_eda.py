import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage4")

# Configurar estilo visual científico
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['grid.color'] = '#DDDDDD'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

def save_plot(name: str):
    """Guarda el gráfico actual en alta resolución (PNG 300 dpi y PDF vectorial)"""
    png_path = os.path.join(config.FIGURES_DIR, f"{name}.png")
    pdf_path = os.path.join(config.FIGURES_DIR, f"{name}.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico guardado: {png_path} y {pdf_path}")

def run_stage4(df: pd.DataFrame) -> dict:
    """
    Etapa 4: Análisis Exploratorio de Datos (EDA)
    - Calcula descriptivos univariados y tabulaciones cruzadas.
    - Genera gráficos científicos numerados a 300 dpi (PNG/PDF):
      1. Histograma de Edades con KDE
      2. Boxplot de Edad por Sexo de la Víctima
      3. Violin plot de Edad por Tipo de Arma
      4. Gráfico de barras de Homicidios por Tipo de Arma
      5. Gráfico de barras de Homicidios por Presunto Móvil
      6. Gráfico de barras de Homicidios por Distrito (Top 10)
      7. Gráfico de barras de Homicidios por Circuito (Top 15)
      8. Gráfico de torta (Pie chart) por Sexo de la víctima
      9. Serie de tiempo mensual de homicidios (2024-2026)
      10. Heatmap de Homicidios (Día de la semana vs Hora)
    - Retorna diccionario de estadísticas descriptivas resumen.
    """
    logger.info("Iniciando ETAPA 4: Análisis Exploratorio de Datos (EDA)")
    
    # Paleta de colores consistente
    colors_crim = ['#800000', '#A52A2A', '#D2691E', '#CD853F', '#F4A460']
    
    # 1. Gráfico 1: Histograma de Edades con KDE
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Edad_Limpia', kde=True, color='#800000', bins=30, edgecolor='black', alpha=0.7)
    plt.title("Figura 1: Distribución de la Edad de las Víctimas")
    plt.xlabel("Edad (Años)")
    plt.ylabel("Frecuencia (Conteo de Casos)")
    plt.grid(True, axis='y')
    save_plot("Figura1_histograma_edad")
    
    # 2. Gráfico 2: Boxplot de Edad por Sexo
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df[df[config.COL_SEXO].isin(['HOMBRE', 'MUJER'])], x=config.COL_SEXO, y='Edad_Limpia', palette='Set2')
    plt.title("Figura 2: Distribución de Edad por Sexo de la Víctima")
    plt.xlabel("Sexo de la Víctima")
    plt.ylabel("Edad (Años)")
    plt.grid(True, axis='y')
    save_plot("Figura2_boxplot_edad_sexo")
    
    # 3. Gráfico 3: Violin plot de Edad por Tipo de Arma
    plt.figure(figsize=(9, 5))
    sns.violinplot(data=df, x='Grupo_Arma', y='Edad_Limpia', palette='Set3', split=True, inner="quart")
    plt.title("Figura 3: Distribución de Edad según el Tipo de Arma Utilizada")
    plt.xlabel("Tipo de Arma")
    plt.ylabel("Edad (Años)")
    plt.xticks(rotation=15)
    plt.grid(True, axis='y')
    save_plot("Figura3_violin_edad_arma")
    
    # 4. Gráfico 4: Gráfico de barras de Homicidios por Grupo de Arma
    plt.figure(figsize=(8, 5))
    arma_counts = df['Grupo_Arma'].value_counts()
    sns.barplot(x=arma_counts.index, y=arma_counts.values, palette='Reds_r', edgecolor='black')
    plt.title("Figura 4: Frecuencia de Homicidios por Tipo de Arma")
    plt.xlabel("Tipo de Arma")
    plt.ylabel("Conteo de Homicidios")
    plt.grid(True, axis='y')
    save_plot("Figura4_barras_grupo_arma")
    
    # 5. Gráfico 5: Gráfico de barras de Homicidios por Presunto Móvil
    plt.figure(figsize=(9, 6))
    motive_counts = df[config.COL_PRESUN_MOTIVACION].value_counts().head(8)
    sns.barplot(y=motive_counts.index, x=motive_counts.values, palette='Oranges_r', edgecolor='black')
    plt.title("Figura 5: Principales Presuntos Móviles de Homicidio")
    plt.xlabel("Conteo de Homicidios")
    plt.ylabel("Presunto Móvil")
    plt.grid(True, axis='x')
    save_plot("Figura5_barras_movil")
    
    # 6. Gráfico 6: Homicidios por Distrito (Top 10)
    plt.figure(figsize=(9, 5))
    dist_counts = df[config.COL_DISTRITO].value_counts().head(10)
    sns.barplot(x=dist_counts.values, y=dist_counts.index, palette='Purples_r', edgecolor='black')
    plt.title("Figura 6: Top 10 Distritos de Guayaquil con Mayor Cantidad de Homicidios")
    plt.xlabel("Conteo de Homicidios")
    plt.ylabel("Distrito Metropolitano")
    plt.grid(True, axis='x')
    save_plot("Figura6_barras_distrito")
    
    # 7. Gráfico 7: Homicidios por Circuito (Top 15)
    plt.figure(figsize=(10, 6))
    circ_counts = df[config.COL_CIRCUITO].value_counts().head(15)
    sns.barplot(x=circ_counts.values, y=circ_counts.index, palette='GnBu_r', edgecolor='black')
    plt.title("Figura 7: Top 15 Circuitos de Guayaquil con Mayor Cantidad de Homicidios")
    plt.xlabel("Conteo de Homicidios")
    plt.ylabel("Circuito")
    plt.grid(True, axis='x')
    save_plot("Figura7_barras_circuito")
    
    # 8. Gráfico 8: Gráfico de torta (Pie chart) por Sexo
    plt.figure(figsize=(6, 6))
    sex_counts = df[config.COL_SEXO].value_counts()
    plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', colors=['#4682B4', '#FF69B4', '#D3D3D3'],
            startangle=90, wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
    plt.title("Figura 8: Distribución de Víctimas por Sexo")
    save_plot("Figura8_pie_sexo")
    
    # 9. Gráfico 9: Serie de tiempo mensual
    plt.figure(figsize=(10, 5))
    # Crear índice año-mes ordenado
    df['Anio_Mes_Dt'] = pd.to_datetime(df['Anio'].astype(str) + '-' + df['Mes_Num'].astype(str) + '-01')
    ts_data = df.groupby('Anio_Mes_Dt').size().reset_index(name='homicidios')
    plt.plot(ts_data['Anio_Mes_Dt'], ts_data['homicidios'], marker='o', color='#800000', linewidth=2)
    plt.title("Figura 9: Evolución Mensual de Homicidios en Guayaquil (2024-2026)")
    plt.xlabel("Período (Mes-Año)")
    plt.ylabel("Número de Homicidios")
    plt.grid(True)
    save_plot("Figura9_serie_tiempo_mensual")
    
    # 10. Gráfico 10: Heatmap de Día de la Semana vs Hora
    plt.figure(figsize=(10, 6))
    # Ordenar días
    dia_order = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO', 'DOMINGO']
    heatmap_data = df.groupby(['DiaSemana', 'Hora']).size().unstack(fill_value=0)
    # Reordenar filas
    heatmap_data = heatmap_data.reindex(dia_order)
    sns.heatmap(heatmap_data, cmap='Reds', annot=False, fmt='d', cbar_kws={'label': 'Número de Casos'})
    plt.title("Figura 10: Concentración de Homicidios por Día de la Semana y Hora del Hecho")
    plt.xlabel("Hora del Día (Formato 24h)")
    plt.ylabel("Día de la Semana")
    save_plot("Figura10_heatmap_dia_hora")

    # Estadísticos descriptivos numéricos estructurados
    stats = {
        "edad_mean": float(df['Edad_Limpia'].mean()),
        "edad_median": float(df['Edad_Limpia'].median()),
        "edad_std": float(df['Edad_Limpia'].std()),
        "edad_p25": float(df['Edad_Limpia'].quantile(0.25)),
        "edad_p75": float(df['Edad_Limpia'].quantile(0.75)),
        "homicidios_totales": len(df),
        "frec_sexo": df[config.COL_SEXO].value_counts().to_dict(),
        "pct_sexo": (df[config.COL_SEXO].value_counts(normalize=True) * 100).to_dict(),
        "frec_arma": df['Grupo_Arma'].value_counts().to_dict(),
        "pct_arma": (df['Grupo_Arma'].value_counts(normalize=True) * 100).to_dict(),
        "frec_movil": df[config.COL_PRESUN_MOTIVACION].value_counts().to_dict(),
        "pct_movil": (df[config.COL_PRESUN_MOTIVACION].value_counts(normalize=True) * 100).to_dict()
    }
    
    # Guardar resumen en archivo JSON de datos descriptivos para uso rápido
    pd.Series(stats).to_json(os.path.join(config.CLEAN_DATA_DIR, "eda_resumen_estadisticos.json"))
    
    logger.info("ETAPA 4: Análisis Exploratorio de Datos (EDA) finalizado con éxito.")
    return stats
