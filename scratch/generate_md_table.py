import pandas as pd
import os

def main():
    csv_path = r"outputs/data/resultados_riesgo_relativo.csv"
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    md_content = """# Tabla Completa de Riesgo Relativo (RR) por Subcircuito (Guayaquil)

Esta tabla muestra la estimación bayesiana del **Riesgo Relativo (RR)** mediano posterior y su clasificación en quintiles para los **244 subcircuitos** del cantón Guayaquil. Los subcircuitos están ordenados de mayor a menor nivel de riesgo.

### 📊 Tabla de Riesgos Relativos

| N° | Subcircuito | Código | Distrito | Circuito | Homicidios Totales | Riesgo Relativo (RR) | Categoría de Riesgo (Quintiles) | P(RR > 1.5) |
| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    
    for idx, row in df.iterrows():
        # Handle cases where name or district might be null (though they shouldn't be)
        name = row.get('subcircuito_nom', 'N/A')
        code = row.get('cod_subcircu', 'N/A')
        distrito = row.get('distrito_nom', 'N/A')
        circuito = row.get('circuito_nom', 'N/A')
        
        # Fill missing values for subcircuits with 0 homicides
        homicidios = row.get('homicidios_totales', 0.0)
        homicidios_val = int(homicidios) if pd.notnull(homicidios) else 0
        
        rr_median = row.get('rr_estimado_mediana', 1.0)
        p_exc = row.get('probabilidad_excedencia', 0.0)
        categoria = row.get('rr_categoria', 'N/A')
        
        md_content += f"| {idx+1} | {name} | `{code}` | {distrito} | {circuito} | {homicidios_val} | **{rr_median:.2f}** | {categoria} | {p_exc:.3f} |\n"

    # Save to artifact folder
    artifact_path = r"C:\Users\ASUS\.gemini\antigravity-ide\brain\8000dc29-2b20-4530-8269-0f2ed3f93f26\tabla_riesgo_relativo_completa.md"
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Artifact table created successfully at: {artifact_path}")

if __name__ == "__main__":
    main()
