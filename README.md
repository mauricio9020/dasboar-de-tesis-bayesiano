# Análisis Bayesiano de Homicidios en el Cantón Guayaquil mediante Análisis Espacio-Temporal

Repositorio de código, pipeline analítico y tablero de visualización interactivo correspondiente a la tesis de pregrado (Ingeniería en Estadística):
**"Análisis Bayesiano de Homicidios en el Cantón Guayaquil mediante Análisis Espacio-Temporal"** (Guayaquil, Ecuador).

---

## 🚀 Requisitos del Sistema y Entorno

- **Versión de Python recomendada:** Python 3.10 a 3.14 (probado y validado en Windows 64-bit).
- **Compilador C++:** Para el óptimo desempeño del motor simbólico `PyTensor` en PyMC (NUTS MCMC), se incluye autodetección de compilador `g++` (MinGW-w64).
- **Dependencias principales:** `pymc==5.27.0`, `arviz==0.23.0`, `geopandas==1.1.2`, `libpysal==4.15.0`, `streamlit==1.55.0`.

### Instalación de dependencias
```bash
pip install -r requirements.txt
```

---

## 🖥️ Ejecución del Proyecto

### 1. Ejecutar el Pipeline Completo de Procesamiento y Modelado
Ejecuta secuencialmente las 13 etapas metodológicas: desde la limpieza de datos hasta el ajuste de los 4 modelos bayesianos con NUTS, diagnósticos de convergencia y exportación de reportes:
```bash
python run_pipeline.py
```
*Tiempo estimado de ejecución completa:* ~8-15 minutos (4 modelos bayesianos con 4 cadenas MCMC x 2,000 iteraciones cada uno con aceleración C++ vía PyTensor).

### 2. Ejecutar el Tablero Interactivo (Dashboard Streamlit)
```bash
streamlit run dashboard.py
```

---

## 📐 Decisiones Metodológicas y Correcciones Rigurosas

En cumplimiento de las normas de integridad científica y rigor estadístico, se corrigieron los siguientes aspectos metodológicos:

### 1. Eliminación Total de Resultados Fabricados
- **Cero multiplicadores o valores sintéticos:** Se eliminaron las tablas simuladas por factores fijos (`waic * 2.0`, `waic * 1.5`) y `p_WAIC` escritos a mano. La tabla comparativa oficial (`tabla_comparacion_modelos_bayesiana.xlsx`) se genera exclusivamente a partir de ajustes reales con MCMC y validación cruzada `az.compare(..., ic='loo', scale='log')`.
- **Diagnósticos reales:** Se eliminó cualquier gráfico simulado. Si una métrica presenta advertencias (como el diagnóstico de Pareto-$k$), el sistema lo reporta explícitamente sin enmascarar datos.
- **Rango completo en PPC:** El chequeo predictivo posterior (`Figura19_bayesian_ppc.png`) visualiza toda la cola empírica de la distribución sin recortes artificiales.
- **Escala de Criterios Unificada:** Todas las métricas de información predictiva (WAIC y LOO) se calculan y reportan en escala logarítmica (ELPD - Expected Log Predictive Density).
- **Modelos Clásicos Frecuentistas:** En `tabla_modelos_clasicos.xlsx` se eliminó el Pseudo $R^2$ de McFadden para evitar comparaciones no válidas entre Poisson y Binomial Negativa; se conservan exclusivamente Log-Likelihood, AIC y BIC.

### 2. Depuración de Datos y Consistencia de la Muestra
- **Conteo de Casos:**
  - Registros crudos en base policial: 4,716
  - Registros duplicados eliminados: 35
  - Registros fuera del bounding box cantonal: 47
  - **Muestra depurada final:** **N = 4,634** homicidios.
- **Distribución por Sexo:** 4,227 hombres (91.22%), 382 mujeres (8.24%), 25 no determinados (0.54%).
- **Tratamiento de la Edad:** 232 registros con edad no consignada en el parte policial quedan formalmente como `NaN` y se excluyen de las estadísticas descriptivas etarias. Los estadísticos se reportan estrictamente sobre la submuestra válida de **n = 4,402** casos, sin imputaciones artificiales.
- **Variable Antecedentes Penales:** Se encuentra vacía en el 100% de los registros de la fuente policial oficial (4,634 registros nulos); fue excluida formalmente del análisis cuantitativo.
- **Discrepancia de Subcircuitos (244 vs 225):** El shapefile oficial de Guayaquil contiene **244 subcircuitos**. De ellos, **225** registraron al menos 1 homicidio en el período, mientras que **19 subcircuitos** registraron 0 casos. Todos los 244 subcircuitos se mantienen en la matriz de contigüidad y en el panel espacio-temporal con conteo 0 para evitar agujeros y desconexiones en la topología de la vecindad.
- **Tratamiento Temporal Anual (2024-2025):** Con solo 2 años disponibles ($T=2$), una caminata aleatoria de primer orden (RW1) no es paramétricamente identificable frente al intercepto (generando varianza temporal no confinada $\sigma_t$ y divergencias NUTS). Por ello, el modelo espacio-temporal utiliza un **efecto fijo anual** $\beta_{\text{year}} \sim \text{Normal}(0, 1)$ sobre el indicador del año 2025 frente a la línea base 2024, garantizando convergencia limpia y 0 divergencias. El año 2026 (solo enero) se preserva en el panel mensual de 25 meses.

### 3. Matriz Espacial y Grafo Conexo (ESDA)
- **Queen de 1er Orden:** Se utiliza contigüidad espacial Queen física de orden 1 en lugar de k-NN. La matriz Queen respeta la frontera física compartida y los vértices de polígonos irregulares del tejido urbano.
- **Corrección de Densidad de Matriz:** Se corrigió el cálculo de densidad en la tabla ESDA eliminando la doble multiplicación por 100 sobre `w.pct_nonzero`, reportando el valor empírico exacto (~2.15% con promedio de 5.24 vecinos por subcircuito).
- **Conexión de Islas:** Los 4 subcircuitos insulares de Isla Puná (Puná 1, 2, 3, 4) se conectan formalmente a su centroide continental más cercano. Esto garantiza un grafo con 0 islas y un único componente conexo ($n_{\text{comp}} = 1$), condición matemática necesaria para la restricción de suma cero (`sum-to-zero`) en los modelos ICAR/BYM2.
- **Reproducibilidad:** Se fija la semilla `RANDOM_SEED = 42` en todas las etapas estocásticas (Moran con 9,999 permutaciones y muestreo PyMC).

### 4. Diagnóstico de Sobredispersión
- **Test de Cameron & Trivedi (1990):** Calculado y exportado en `tabla_sobredispersion.xlsx`, confirmando rechazo masivo de la hipótesis nula de equidispersión de Poisson.
- **Contraste Frecuentista Clásico:** Comparación Poisson vs Binomial Negativa (AIC 29,243 vs 3,267 con offset; 7,107 vs 3,114 con covariable de área) como justificación empírica de la verosimilitud Binomial Negativa.

### 5. Formulación y Comparación de Modelos Bayesianos (Stage 8)
- **Exposición:** Ante la ausencia de proyecciones demográficas oficiales a escala subcircuital, el área territorial se modela como covariable $\beta_{\text{area}} \log(\text{Area}_i)$ con su propio coeficiente estimado (prior $\mathcal{N}(0, 1)$), en vez de forzar un offset con elasticidad unitaria fija.
- **Prior Espacial ICAR:** Se utiliza la formulación canónica `pm.ICAR` de PyMC (Morris et al., 2019; Besag et al., 1991) con parametrización no centrada.
- **Muestreador:** NUTS (No-U-Turn Sampler) con 4 cadenas, tune = 1,000, draws = 1,000, `target_accept = 0.99`.
- **Cuatro Modelos Ajustados y Comparados con LOO-CV:**
  1. *Poisson Jerárquico:* ICAR + Efecto Fijo Anual.
  2. *Binomial Negativa Jerárquico:* ICAR + Efecto Fijo Anual.
  3. *Binomial Negativa con ICAR + Efecto No Estructurado (Besag et al., 1991).*
  4. *BYM2 Binomial Negativo (Prior Beta(0.5, 0.5)):* parámetro de mezcla $\rho \sim \text{Beta}(0.5, 0.5)$ (Riebler et al., 2016).
- **Equivalencia Predictiva:** La diferencia de ELPD entre el modelo (c) y el modelo (d) es de solo 0.10 puntos con error estándar $dse = 1.18$. Al no ser estadísticamente significativa, se concluye formalmente que ambos modelos tienen desempeño equivalente y que BYM2 no supera a Besag.
- **Lectura del Parámetro $\rho$:** Se interpreta a través de su intervalo de credibilidad posterior (media 0.81-0.82, 95% HDI [0.64, 1.00]), documentando que la verosimilitud no logra discriminar tajantemente entre varianza espacial pura y total.
- **Diagnóstico Pareto-$k$:** Se registra el desglose completo de observaciones por rango de $k$ y se explicita la advertencia metodológica inherente a modelos jerárquicos espaciales latentes con $T=2$ en unidades pequeñas.

### 6. Definición Rigurosa del Riesgo Relativo (RR)
- Se estandariza formalmente muestra a muestra sobre la media espacial cantonal:
  $$RR_i^{(s)} = \frac{\exp(\phi_i^{(s)})}{\frac{1}{N} \sum_{k=1}^N \exp(\phi_k^{(s)})}$$
- Esto garantiza que el riesgo promedio cantonal de referencia sea exactamente 1.00. Un subcircuito con $RR = 1.5$ presenta un 50% más de riesgo que el promedio cantonal.
- La probabilidad de excedencia $P(RR_i > 1.5)$ cuantifica la certidumbre posterior de riesgo crítico sobre la referencia cantonal estandarizada.

---

## 📁 Estructura del Repositorio

```text
├── dashboard.py                  # Tablero interactivo en Streamlit
├── run_pipeline.py               # Orquestador del pipeline científico
├── requirements.txt              # Dependencias con versiones fijadas
├── README.md                     # Documentación técnica y metodológica
├── Muertes_Violentas_...xlsx     # Base de datos policial cruda oficial
├── data/
│   ├── Subcircuitos/             # Capa geográfica oficial (Shapefile 244 subcircuitos)
│   └── clean_data/               # Datos normalizados, GeoJSON y paneles agregados
├── outputs/
│   ├── figures/                  # Figuras científicas numeradas (PNG 300 dpi y PDF vectorial)
│   ├── tables/                   # Tablas analíticas en Excel (ESDA, Diagnósticos, LOO-CV)
│   ├── models/                   # Trazas de inferencia ArviZ NetCDF (.nc)
│   └── reports/                  # Reporte integrado en formato Word (.docx)
├── scratch/                      # Scripts auxiliares, compilador y borradores de trabajo
└── src/
    ├── config.py                 # Rutas relativas pathlib y configuración global
    ├── stage1_loader.py          # Carga e integridad de datos
    ├── stage2_cleaner.py         # Depuración y control de calidad
    ├── stage3_engineering.py     # Ingeniería de variables y paneles espaciales
    ├── stage4_eda.py             # Análisis exploratorio de datos
    ├── stage5_esda.py            # ESDA, matriz Queen conexa y Moran's I
    ├── stage6_overdispersion.py  # Diagnóstico formal Cameron-Trivedi
    ├── stage7_classical.py       # Modelos GLM frecuentistas
    ├── stage8_bayesian.py        # Modelos Bayesianos NUTS (4 modelos + LOO-CV)
    ├── stage9_diagnostics.py     # Diagnósticos ArviZ, R_hat, ESS, PPC completo
    ├── stage10_relative_risk.py  # Estimación calibrada de RR y mapas
    ├── stage11_space_time_dynamics.py # Análisis espacio-temporal de hotspots
    └── stage13_exporter.py       # Exportación automática a documento Word
```
