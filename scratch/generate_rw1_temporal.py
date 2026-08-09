import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Add current dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def main():
    print("Cargando traza bayesiana...")
    trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    if not os.path.exists(trace_path):
        print(f"Error: No existe la traza en {trace_path}")
        return
    trace = az.from_netcdf(trace_path)
    
    # Extract posterior samples of temporal effect 'gamma'
    # Shape of posterior['gamma'] is (chains, draws, periodos)
    gamma_samples = trace.posterior['gamma'].values.reshape(-1, trace.posterior['gamma'].shape[-1])
    n_samples, n_periodos = gamma_samples.shape
    print(f"Muestras de la posterior de gamma cargadas: {n_samples} draws para {n_periodos} períodos.")
    
    # Compute stats
    gamma_mean = np.mean(gamma_samples, axis=0)
    gamma_median = np.median(gamma_samples, axis=0)
    gamma_lower = np.percentile(gamma_samples, 2.5, axis=0)
    gamma_upper = np.percentile(gamma_samples, 97.5, axis=0)
    
    # Years mapping (from datos_limpios / stage8_bayesian)
    years = [2024, 2025, 2026]
    if n_periodos != len(years):
        print(f"Advertencia: El número de períodos en el modelo ({n_periodos}) difiere de los años esperados ({len(years)})")
        years = list(range(1, n_periodos + 1))
        
    df_temporal = pd.DataFrame({
        'Año': years,
        'Efecto_Temporal_Media': gamma_mean,
        'Efecto_Temporal_Mediana': gamma_median,
        'HDI_inf_95': gamma_lower,
        'HDI_sup_95': gamma_upper
    })
    
    # Save CSV and Excel
    csv_out_path = os.path.join(config.CLEAN_DATA_DIR, "tendencia_temporal_rw1.csv")
    excel_out_path = os.path.join(config.TABLES_DIR, "tabla_tendencia_temporal_rw1.xlsx")
    
    df_temporal.to_csv(csv_out_path, index=False)
    df_temporal.to_excel(excel_out_path, index=False)
    print(f"[OK] Tabla de tendencia guardada en CSV: {csv_out_path}")
    print(f"[OK] Tabla de tendencia guardada en Excel: {excel_out_path}")
    
    # Print console table
    print("\nEstadísticas del Efecto Temporal Caminata Aleatoria (RW1):")
    print(df_temporal.to_string(index=False))
    
    # Generate Markdown Table Artifact
    md_table = """# Tabla de Tendencia Temporal del Riesgo (Caminata Aleatoria RW1)

Esta tabla contiene las estimaciones bayesianas del componente temporal del riesgo, modelado mediante una **Caminata Aleatoria de Orden 1 (RW1)** ($\gamma_t$) para el período 2024-2026. 
Un incremento en el valor de $\gamma_t$ representa una tendencia al alza en el riesgo general de homicidios en todo el cantón Guayaquil, independiente del efecto espacial.

### 📊 Valores del Efecto Temporal (RW1)

| Año | Efecto Temporal (Media posterior) | Efecto Temporal (Mediana posterior) | Intervalo de Credibilidad (HDI 95%) | Tendencia Relativa |
| :---: | :---: | :---: | :---: | :---: |
"""
    for idx, row in df_temporal.iterrows():
        yr = int(row['Año'])
        mean_val = row['Efecto_Temporal_Media']
        median_val = row['Efecto_Temporal_Mediana']
        lower = row['HDI_inf_95']
        upper = row['HDI_sup_95']
        
        # Calculate trend direction
        if idx == 0:
            trend = "Línea base inicial"
        else:
            prev_val = df_temporal.iloc[idx-1]['Efecto_Temporal_Mediana']
            diff = median_val - prev_val
            trend = f"📈 Alza ({diff:+.4f})" if diff > 0 else f"📉 Baja ({diff:+.4f})"
            
        md_table += f"| {yr} | {mean_val:.4f} | **{median_val:.4f}** | [{lower:.4f}, {upper:.4f}] | {trend} |\n"
        
    artifact_table_path = r"C:\Users\ASUS\\.gemini\antigravity-ide\brain\8000dc29-2b20-4530-8269-0f2ed3f93f26\tabla_tendencia_temporal_rw1.md"
    with open(artifact_table_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"[OK] Tabla markdown guardada en: {artifact_table_path}")
    
    # -------------------------------------------------------------------------
    # GENERACIÓN DE GRÁFICO DE TENDENCIA (Matplotlib)
    # -------------------------------------------------------------------------
    print("Generando gráfico de tendencia de la Caminata Aleatoria...")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Plot 95% HDI shaded area
    ax.fill_between(
        years, gamma_lower, gamma_upper,
        color='#2196f3',
        alpha=0.15,
        label='Intervalo de Credibilidad (HDI 95%)'
    )
    
    # Plot median line
    ax.plot(
        years, gamma_median,
        color='#0d47a1',
        linewidth=2.5,
        marker='o',
        markersize=8,
        label='Efecto Temporal Posterior (Mediana)'
    )
    
    # Add values text next to markers
    for x, y in zip(years, gamma_median):
        ax.text(
            x, y + 0.05, f"{y:+.3f}",
            fontsize=10,
            fontweight='bold',
            color='#0d47a1',
            ha='center',
            va='bottom'
        )
        
    # Style axes
    ax.set_xticks(years)
    ax.set_xticklabels([str(yr) for yr in years], fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    
    # Labels and titles
    ax.set_xlabel('Año', fontsize=12, fontweight='bold', labelpad=12)
    ax.set_ylabel('Efecto Temporal ($\gamma_t$ en escala log)', fontsize=12, fontweight='bold', labelpad=12)
    ax.set_title('Evolución Temporal del Riesgo: Caminata Aleatoria de Orden 1 (RW1)\nModelo Bayesiano de Homicidios en Guayaquil (2024-2026)', fontsize=13, fontweight='bold', pad=15)
    
    # Horizontal line at 0 for reference
    ax.axhline(0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Grid lines
    ax.grid(color='#eeeeee', linestyle='-', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#cccccc')
    
    # Borders
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#aaaaaa')
        
    # Save figure
    png_path = os.path.join(config.FIGURES_DIR, "tendencia_temporal_rw1.png")
    pdf_path = os.path.join(config.FIGURES_DIR, "tendencia_temporal_rw1.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico de tendencia guardado en PNG: {png_path}")

if __name__ == "__main__":
    main()
