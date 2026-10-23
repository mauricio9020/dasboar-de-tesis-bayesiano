import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import logging
from scipy.stats import nbinom
from shapely.geometry import Point
import arviz as az
import matplotlib.pyplot as plt

# Agregar ruta actual para importar configuración
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config

# Configuración de Página
st.set_page_config(
    page_title="Tesis UTM - Modelado Bayesiano",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados Premium (Tema Oscuro Tecnológico)
st.markdown("""
    <style>
    /* Ocultar menú y ajustar padding */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Variables de diseño */
    :root {
        --bg-panel: #161b22;
        --border-color: #30363d;
        --text-main: #c9d1d9;
        --danger-color: #ff4444;
        --warning-color: #ffb74d;
        --success-color: #66bb6a;
        --accent-color: #58a6ff;
    }
    
    /* Contenedor de métricas */
    .stat-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 18px;
        color: var(--text-main);
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transition: all 0.2s ease-in-out;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
        border-color: var(--accent-color);
    }
    .stat-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--accent-color);
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    .badge {
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-red { background-color: rgba(255, 68, 68, 0.15); color: var(--danger-color); }
    .badge-green { background-color: rgba(102, 187, 106, 0.15); color: var(--success-color); }
    .badge-orange { background-color: rgba(255, 183, 77, 0.15); color: var(--warning-color); }
    
    .progress-track {
        height: 8px;
        background: linear-gradient(to right, var(--success-color), var(--warning-color), var(--danger-color));
        border-radius: 4px;
        margin: 15px 0 10px 0;
        position: relative;
    }
    
    /* Ecuaciones en Látex */
    .latex-container {
        background-color: #0d1117;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
    }
    
    /* Estilos personalizados para chips de multiselect y selectbox */
    div[data-baseweb="tag"] {
        background-color: #ff4444 !important;
        color: white !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="tag"] span {
        color: white !important;
    }
    div[data-baseweb="tag"] button {
        color: white !important;
    }
    div[data-baseweb="tag"] button:hover {
        color: #ffb74d !important;
        background-color: transparent !important;
    }
    div[data-baseweb="tag"]:hover {
        background-color: #ff3333 !important;
    }
    div[data-baseweb="select"] {
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- CARGA DE DATOS ----------------- #
def get_rho_advi_nuts_str():
    nuts_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_advi_nuts.xlsx")
    if os.path.exists(nuts_path):
        try:
            df_temp = pd.read_excel(nuts_path)
            row_rho = df_temp[df_temp.iloc[:, 1].str.contains('rho', case=False, na=False)]
            if not row_rho.empty:
                return f"{row_rho['Media_ADVI'].values[0]:.4f}", f"{row_rho['Media_NUTS'].values[0]:.4f}"
        except Exception:
            pass
    return "0.9998", "0.8199"

@st.cache_data
def load_clean_data():
    clean_csv_path = os.path.join(config.CLEAN_DATA_DIR, "datos_limpios.csv")
    if os.path.exists(clean_csv_path):
        df = pd.read_csv(clean_csv_path)
        df['fecha_infraccion_dt'] = pd.to_datetime(df['fecha_infraccion_dt'])
        return df
    else:
        # Fallback a datos crudos si no se ha ejecutado el pipeline
        if os.path.exists(config.EXCEL_PATH):
            df = pd.read_excel(config.EXCEL_PATH)
            # Limpieza básica para que el dashboard no crashee
            df['latitud'] = pd.to_numeric(df[config.COL_COORD_Y].astype(str).str.replace(',', '.'), errors='coerce')
            df['longitud'] = pd.to_numeric(df[config.COL_COORD_X].astype(str).str.replace(',', '.'), errors='coerce')
            df = df.dropna(subset=['latitud', 'longitud'])
            df['fecha_infraccion_dt'] = pd.to_datetime(df[config.COL_FECHA], errors='coerce')
            df['Anio'] = df['fecha_infraccion_dt'].dt.year
            df['Mes_Num'] = df['fecha_infraccion_dt'].dt.month
            df['Hora'] = 12
            df['Grupo_Edad'] = "18-29 ANOS"
            df['Grupo_Arma'] = "ARMA DE FUEGO"
            df['Tipo_Muerte'] = "ASESINATO"
            return df
        else:
            return None

@st.cache_data
def load_spatial_data():
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
    if os.path.exists(geojson_path):
        return gpd.read_file(geojson_path)
    
    # Intentar cargar GeoJSON básico ESDA
    esda_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_esda.geojson")
    if os.path.exists(esda_path):
        return gpd.read_file(esda_path)
        
    # Cargar shapefile directo y proyectarlo a EPSG:4326
    if os.path.exists(config.SHAPEFILE_PATH):
        gdf = gpd.read_file(config.SHAPEFILE_PATH)
        gdf = gdf[gdf['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
        gdf = gdf.rename(columns={'COD_SUBCIR': 'cod_subcircu', 'NAM_SUBCIR': 'subcircuito_nom', 'NAM_CIRCUI': 'circuito_nom', 'NAM_DISTRI': 'distrito_nom'})
        gdf['Area_km2'] = gdf['Shape_Area'] / 1e6
        return gdf.to_crs("EPSG:4326")
        
    return None

@st.cache_data
def load_annual_risk_data():
    danual_path = os.path.join(config.CLEAN_DATA_DIR, "datos_agregados_anual.csv")
    dhotspots_path = os.path.join(config.CLEAN_DATA_DIR, "dinamica_hotspots.csv")
    df_anual = pd.read_csv(danual_path) if os.path.exists(danual_path) else None
    df_hotspots = pd.read_csv(dhotspots_path) if os.path.exists(dhotspots_path) else None
    return df_anual, df_hotspots

df_homicidios = load_clean_data()
gdf_subcircuitos = load_spatial_data()
df_anual_map, df_hotspots_map = load_annual_risk_data()

@st.cache_resource
def load_pymc_trace():
    trace_path = os.path.join(config.MODELS_DIR, "pymc_trace.nc")
    if os.path.exists(trace_path):
        import shutil
        temp_trace_path = os.path.join(config.MODELS_DIR, "pymc_trace_temp_dashboard.nc")
        try:
            shutil.copy2(trace_path, temp_trace_path)
            return az.from_netcdf(temp_trace_path)
        except Exception:
            try:
                return az.from_netcdf(trace_path)
            except Exception as e:
                import logging
                logging.warning(f"No se pudo cargar la traza bayesiana desde NetCDF: {e}")
                return None
    return None

trace = load_pymc_trace()

# Cargar estimaciones de riesgo relativo reales desde el archivo de salida del pipeline
rr_csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
if os.path.exists(rr_csv_path) and gdf_subcircuitos is not None:
    try:
        df_rr_res = pd.read_csv(rr_csv_path)
        rr_cols_to_merge = [c for c in ['cod_subcircu', 'rr_estimado_mediana', 'rr_estimado_media', 'rr_limite_inf_95', 'rr_limite_sup_95', 'probabilidad_excedencia', 'prob_exced_1', 'prob_segura'] if c in df_rr_res.columns]
        # Si ya existen en gdf_subcircuitos, no duplicar
        cols_missing = [c for c in rr_cols_to_merge if c not in gdf_subcircuitos.columns or c == 'cod_subcircu']
        if len(cols_missing) > 1:
            gdf_subcircuitos = gdf_subcircuitos.merge(df_rr_res[cols_missing], on='cod_subcircu', how='left')
    except Exception as e:
        st.warning(f"Aviso al cargar resultados de riesgo relativo: {e}")

# Validar que los archivos existan
if df_homicidios is None or gdf_subcircuitos is None:
    st.error("⚠️ Error: No se encontraron las bases de datos. Asegúrate de colocar el archivo Excel de muertes violentas en el directorio del proyecto y ejecutar el pipeline.")
    st.stop()

# ----------------- SIDEBAR CONTROLS ----------------- #
with st.sidebar:
    st.title("🛡️ Tesis UTM")
    st.markdown("---")
    
    st.subheader("🛠️ Panel de Control")
    umbral_riesgo = st.slider("Umbral de Alerta de Riesgo Relativo P(RR > x)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    
    opcion_mapa = st.selectbox(
        "¿Qué capa deseas visualizar en el mapa?",
        ("Probabilidad de Excedencia", "Riesgo Relativo Estimado", "Conteo de Homicidios")
    )
    
    st.markdown("---")
    st.caption("Tesis de Grado: *Ingeniería en Estadística - UTM*")
    st.caption("Cantón Guayaquil - Período de Análisis: 2024-2026")

# ----------------- CUERPO PRINCIPAL ----------------- #
st.title("🛡️ Tesis UTM: Modelado Bayesiano Espacio-Temporal")
st.markdown("Análisis y visualización de muertes violentas en el cantón Guayaquil (2024 – enero 2026)")

# Tabs del Dashboard
tab_kpi, tab_mapa, tab_bayes, tab_modelo_bayesiano, tab_sim, tab_downloads = st.tabs([
    "📊 Análisis Exploratorio y Preprocesamiento",
    "🗺️ Mapa de Riesgo Bayesiano",
    "⚖️ Evaluación del Modelo Bayesiano",
    "🧠 Modelo Bayesiano",
    "🎯 Simulador Táctico de Riesgo",
    "📥 Descarga de Reportes y Datos"
])

# ================= TAB 1: EDA & KPIs =================
with tab_kpi:
    st.subheader("Análisis Exploratorio y Preprocesamiento de la Base de Muertes Violentas")
    
    st.info("""
    📌 **Avisos Metodológicos de Calidad de Datos:**
    - **Muestra depurada:** **N = 4,634** homicidios confirmados (4,716 registros crudos menos 35 duplicados y 47 fuera del bounding box oficial de Guayaquil).
    - **Año 2026:** Contiene únicamente el mes de **enero (202 casos)**. Para no distorsionar las tasas anuales ni la estimación del efecto temporal, el panel anual del modelo bayesiano modela los años completos **2024 y 2025** (488 observaciones en 244 subcircuitos), preservando 2026 en el panel mensual de 25 meses.
    - **Variable Edad:** Registra 232 valores faltantes en la fuente policial original; las estadísticas de edad se reportan estrictamente sobre la submuestra válida de **n = 4,402** casos sin imputaciones sintéticas.
    - **Variable Antecedentes:** Se encuentra vacía en el 100% de los registros policiales (4,634 registros); fue excluida formalmente del análisis.
    """)
    
    # Contenedor de filtros premium con diseño de tarjeta
    with st.container(border=True):
        st.markdown("<h5 style='margin-top:0px; margin-bottom:10px;'><i class='fa-solid fa-filter'></i> 📊 Filtros de Visualización</h5>", unsafe_allow_html=True)
        col_fil1, col_fil2, col_fil3 = st.columns(3)
        
        with col_fil1:
            years = sorted(df_homicidios['Anio'].dropna().unique())
            periodo_opciones = ["Todo el Período"] + [f"Solo {year}" for year in years]
            periodo_seleccionado = st.selectbox("Filtro de Período", options=periodo_opciones, index=0)
            
        with col_fil2:
            armas = sorted(df_homicidios['Grupo_Arma'].dropna().unique())
            arma_opciones = ["Todas las Armas"] + armas
            arma_seleccionada = st.selectbox("Tipo de Arma", options=arma_opciones, index=0)
            
        with col_fil3:
            sexos = sorted(df_homicidios[config.COL_SEXO].dropna().unique())
            sexo_opciones = ["Todos"] + sexos
            sexo_seleccionado = st.selectbox("Sexo de la Víctima", options=sexo_opciones, index=0)
            
    # Filtrar por año seleccionado
    if periodo_seleccionado == "Todo el Período":
        df_filtered_periodo = df_homicidios.copy()
    else:
        try:
            sel_year = int(periodo_seleccionado.split()[-1])
            df_filtered_periodo = df_homicidios[df_homicidios['Anio'] == sel_year].copy()
        except Exception:
            df_filtered_periodo = df_homicidios.copy()
            
    # Filtrar por sexo seleccionado
    if sexo_seleccionado != "Todos":
        df_filtered_periodo = df_filtered_periodo[df_filtered_periodo[config.COL_SEXO] == sexo_seleccionado].copy()
        
    # Filtrar por tipo de arma seleccionado (para obtener df_filtered final)
    if arma_seleccionada != "Todas las Armas":
        df_filtered = df_filtered_periodo[df_filtered_periodo['Grupo_Arma'] == arma_seleccionada].copy()
    else:
        df_filtered = df_filtered_periodo.copy()
            
    # Métricas clave en tarjetas
    col1, col2, col3, col4 = st.columns(4)
    
    total_homicidios = len(df_filtered)
    total_periodo_sexo = len(df_filtered_periodo)
    
    # Calcular métrica dinámica de arma para resolver el problema de 0% cuando se filtra un tipo específico
    if arma_seleccionada == "Todas las Armas":
        arma_kpi_title = '<i class="fa-solid fa-gun"></i> Uso de Armas de Fuego'
        arma_kpi_value = (df_filtered['Grupo_Arma'] == 'ARMA DE FUEGO').mean() * 100 if total_homicidios > 0 else 0
        arma_kpi_subtitle = "Porcentaje del total filtrado"
    else:
        arma_kpi_title = f'<i class="fa-solid fa-circle-info"></i> Proporción: {arma_seleccionada.title()}'
        arma_kpi_value = (df_filtered_periodo['Grupo_Arma'] == arma_seleccionada).mean() * 100 if total_periodo_sexo > 0 else 0
        arma_kpi_subtitle = "De los homicidios del período/sexo"
        
    # Contar subcircuitos de riesgo muy alto desde CSV aparte
    try:
        csv_rr_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
        df_rr_csv = pd.read_csv(csv_rr_path)
        muy_alto_count = int(df_rr_csv['rr_categoria'].astype(str).str.startswith("Muy Alto").sum())
        valor_riesgo_muy_alto = f"{muy_alto_count} / 244 Zonas"
    except Exception:
        valor_riesgo_muy_alto = "N/D"
    
    # Calcular métricas de edad con detalles del grupo líder y rango
    if total_homicidios > 0:
        mean_age = df_filtered['Edad_Limpia'].mean()
        median_age = df_filtered['Edad_Limpia'].median()
        grupo_counts = df_filtered['Grupo_Edad'].value_counts()
        if len(grupo_counts) > 0:
            top_grupo = grupo_counts.index[0]
            top_grupo_pct = (grupo_counts.values[0] / total_homicidios) * 100
            age_subtitle = f"Mediana: {median_age:.0f} años | Líder: {str(top_grupo).replace('ANOS', 'años')} ({top_grupo_pct:.0f}%)"
        else:
            age_subtitle = f"Mediana: {median_age:.0f} años"
    else:
        mean_age = 0
        age_subtitle = "Sin registros en este período"
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><i class="fa-solid fa-skull"></i> Homicidios Filtrados</div>
            <h2>{total_homicidios:,} Casos</h2>
            <p style="font-size:0.8rem; color:#8b949e;">En el período seleccionado</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">{arma_kpi_title}</div>
            <h2>{arma_kpi_value:.1f}%</h2>
            <p style="font-size:0.8rem; color:#8b949e;">{arma_kpi_subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><i class="fa-solid fa-triangle-exclamation"></i> Riesgo Muy Alto</div>
            <h2>{valor_riesgo_muy_alto}</h2>
            <p style="font-size:0.8rem; color:#8b949e;">Quintil superior del riesgo relativo (RR) (no cambia con los filtros)</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        valid_age_cnt = df_filtered['Edad_Limpia'].dropna().count()
        missing_age_cnt = df_filtered['Edad_Limpia'].isna().sum()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><i class="fa-solid fa-user"></i> Edad Promedio (n = {valid_age_cnt:,})</div>
            <h2>{mean_age:.1f} Años</h2>
            <p style="font-size:0.8rem; color:#8b949e;">{age_subtitle} | {missing_age_cnt} faltantes excluidos</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Evolución Temporal de Casos (Conteo Mensual)**")
        df_filtered['Año_Mes'] = df_filtered['Anio'].astype(str) + "-" + df_filtered['Mes_Num'].astype(str).str.zfill(2)
        ts_df = df_filtered.groupby('Año_Mes').size().reset_index(name='Casos')
        fig_ts = px.line(ts_df, x='Año_Mes', y='Casos', markers=True, color_discrete_sequence=['#ff4444'], labels={'Año_Mes': 'Mes'})
        fig_ts.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
        st.plotly_chart(fig_ts, width="stretch")
        
    with col_g2:
        st.markdown("**Distribución por Presunto Móvil**")
        if total_homicidios > 0:
            mov_counts = df_filtered[config.COL_PRESUN_MOTIVACION].value_counts().reset_index()
            mov_counts.columns = ['Móvil', 'Casos']
            mov_counts['% del total'] = (mov_counts['Casos'] / total_homicidios * 100).map(lambda x: f"{x:.2f}%")
            st.dataframe(mov_counts, hide_index=True, width="stretch")
        else:
            st.info("Sin registros")

    st.markdown("---")
    st.markdown("#### 🧹 Proceso de Limpieza y Distribuciones Geodemográficas")
    
    # Calcular estadísticas dinámicas para la explicación
    if total_homicidios > 0:
        top_mech = df_filtered['Grupo_Arma'].value_counts().index[0]
        top_mech_pct = (df_filtered['Grupo_Arma'].value_counts().values[0] / total_homicidios) * 100
        
        top_dist = df_filtered[config.COL_DISTRITO].value_counts().index[0]
        top_dist_count = df_filtered[config.COL_DISTRITO].value_counts().values[0]
        top_dist_pct = (top_dist_count / total_homicidios) * 100
        
        explicacion_texto = f"""
        <div class="stat-card" style="border-left: 5px solid var(--accent-color); margin-top: 10px; min-height: 380px;">
            <div class="stat-card-title" style="color: var(--accent-color); font-size: 0.95rem;">
                <i class="fa-solid fa-file-invoice"></i> ANÁLISIS DE DISTRIBUCIONES
            </div>
            <p style="font-size: 0.85rem; line-height: 1.4; color: var(--text-main); margin-top: 10px;">
                De acuerdo con la base de datos de muertes violentas filtrada (<b>{total_homicidios} casos</b>):
            </p>
            <ul style="font-size: 0.8rem; line-height: 1.5; color: var(--text-main); padding-left: 15px; margin-top: 5px;">
                <li>El mecanismo de agresión predominante es el <b>{top_mech}</b>, representando el <b>{top_mech_pct:.1f}%</b> de los eventos.</li>
                <li style="margin-top: 8px;">A nivel territorial, el Distrito con mayor concentración de homicidios es <b>{top_dist}</b>, acumulando <b>{top_dist_count} casos</b> (<b>{top_dist_pct:.1f}%</b> del total).</li>
                <li style="margin-top: 8px;">Estos resultados reflejan asimetrías críticas tanto en los recursos bélicos como en la focalización geográfica de los homicidios, sirviendo como fundamento de heterogeneidad para el ajuste del modelo Bayesiano espacial.</li>
            </ul>
        </div>
        """
    else:
        explicacion_texto = """
        <div class="stat-card" style="border-left: 5px solid var(--border-color); margin-top: 10px; min-height: 380px;">
            <div class="stat-card-title"><i class="fa-solid fa-file-invoice"></i> ANÁLISIS DE DISTRIBUCIONES</div>
            <p style="font-size: 0.85rem; color: #8b949e; margin-top: 15px; text-align: center;">
                No hay datos suficientes para generar explicaciones con los filtros seleccionados.
            </p>
        </div>
        """

    col_c1, col_c2, col_c3 = st.columns([2.3, 2.5, 2.0])
    
    with col_c1:
        st.markdown("**Distribución de Mecanismos de Homicidio**")
        if total_homicidios > 0:
            arma_dist = df_filtered['Grupo_Arma'].value_counts().reset_index(name='Casos')
            arma_dist.columns = ['Mecanismo', 'Casos']
            fig_pie = px.pie(arma_dist, values='Casos', names='Mecanismo', hole=0.4, 
                             color_discrete_sequence=['#ff4444', '#ffb74d', '#58a6ff', '#66bb6a', '#c9d1d9'])
            fig_pie.update_layout(
                showlegend=True, 
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#c9d1d9',
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("Sin registros")
            
    with col_c2:
        st.markdown("**Frecuencia de Casos por Distrito**")
        if total_homicidios > 0:
            dist_counts = df_filtered[config.COL_DISTRITO].value_counts().reset_index()
            dist_counts.columns = ['Distrito', 'Casos']
            fig_bar_dist = px.bar(
                dist_counts, 
                x='Casos', 
                y='Distrito', 
                orientation='h', 
                color='Casos', 
                color_continuous_scale='Reds'
            )
            fig_bar_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#c9d1d9',
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=380
            )
            fig_bar_dist.update_yaxes(autorange="reversed", dtick=1)
            st.plotly_chart(fig_bar_dist, width="stretch")
        else:
            st.info("Sin registros")
            
    with col_c3:
        st.markdown("**Interpretación**")
        st.markdown(explicacion_texto, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Distribución Demográfica de Edad y Sexo")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**Distribución de la Edad de las Víctimas**")
        if total_homicidios > 0:
            fig_hist = px.histogram(
                df_filtered, 
                x='Edad_Limpia', 
                nbins=30, 
                color_discrete_sequence=['#800000'],
                labels={'Edad_Limpia': 'Edad (Años)'}
            )
            fig_hist.update_traces(marker_line_color='black', marker_line_width=1)
            fig_hist.update_layout(
                yaxis_title_text='Frecuencia (Conteo de Casos)',
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#c9d1d9',
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_hist, width="stretch")
        else:
            st.info("Sin registros")
            
    with col_d2:
        st.markdown("**Distribución de Edad por Sexo de la Víctima**")
        if total_homicidios > 0:
            df_sex = df_filtered[df_filtered[config.COL_SEXO].isin(['HOMBRE', 'MUJER'])]
            if not df_sex.empty:
                fig_box = px.box(
                    df_sex, 
                    x=config.COL_SEXO, 
                    y='Edad_Limpia', 
                    color=config.COL_SEXO,
                    color_discrete_map={'HOMBRE': '#66c2a5', 'MUJER': '#fc8d62'},
                    labels={'Edad_Limpia': 'Edad (Años)', config.COL_SEXO: 'Sexo de la Víctima'}
                )
                fig_box.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='#c9d1d9',
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_box, width="stretch")
            else:
                st.info("Sin datos de Hombres/Mujeres para graficar boxplot")
        else:
            st.info("Sin registros")
            
    # Sección ESDA: Autocorrelación Espacial Global de Moran
    st.markdown("---")
    st.subheader("📍 Autocorrelación Espacial Global de Moran (ESDA)")
    st.markdown("""
    Se evalúa la dependencia espacial mediante el **Índice de Moran Global** bajo dos especificaciones:
    - **(a) Período del Modelo 2024-2025 (Principal de la Tesis):** Conteo agregado correspondiente a los 2 años completos modelados en el panel bayesiano.
    - **(b) Período Completo 2024-2026 (Exploratorio):** Incluye enero 2026 como contexto del período total registrado.
    
    Ambas versiones utilizan la matriz de contigüidad **Queen de primer orden con las 4 islas de Puná conectadas** a sus costas continentales más próximas, 9,999 permutaciones aleatorias y semilla fija 42.
    """)
    
    esda_excel_path = os.path.join(config.TABLES_DIR, "tabla_esda_moran.xlsx")
    if os.path.exists(esda_excel_path):
        try:
            df_esda_moran = pd.read_excel(esda_excel_path)
            def format_moran_val(x):
                try:
                    f = float(x)
                    if f.is_integer():
                        return str(int(f))
                    return f"{f:.2f}"
                except (ValueError, TypeError):
                    return str(x)
            for c in df_esda_moran.columns[1:]:
                df_esda_moran[c] = df_esda_moran[c].apply(format_moran_val).astype(str)
            st.dataframe(df_esda_moran, hide_index=True, width="stretch")
        except Exception as e:
            st.error(f"Error al cargar tabla ESDA: {e}")
            
    col_esda1, col_esda2 = st.columns(2)
    with col_esda1:
        f11_path = os.path.join(config.FIGURES_DIR, "Figura11_moran_scatterplot.png")
        if os.path.exists(f11_path):
            st.image(f11_path, caption="Figura 11: Moran Scatterplot de Homicidios 2024-2025", width="stretch")
    with col_esda2:
        f12_path = os.path.join(config.FIGURES_DIR, "Figura12_lisa_clusters.png")
        if os.path.exists(f12_path):
            st.image(f12_path, caption="Figura 12: Mapa de Agrupamiento LISA (Homicidios 2024-2025)", width="stretch")

with tab_mapa:
    st.subheader("🗺️ Mapa Probabilístico e Intensidad de Riesgo")
    st.markdown("Este mapa interactivo presenta la distribución del riesgo relativo y las probabilidades de excedencia espacial estimadas mediante el modelo Bayesiano jerárquico ICAR.")
    
    # Ordenar homicidios cronológicamente
    df_homicidios_sorted = df_homicidios.sort_values(by='fecha_infraccion_dt').copy()
    total_casos_base = len(df_homicidios_sorted)
    
    # Filtro de Casos Acumulados con barra deslizadora (slider) de 100 en 100 para ver evolución temporal
    col_fmap1, col_fmap2 = st.columns([2.5, 1.5])
    with col_fmap1:
        map_cases_selected = st.slider(
            "📈 Casos Acumulados Cronológicamente (Desliza para ver la evolución del crecimiento)",
            min_value=100,
            max_value=total_casos_base,
            value=total_casos_base,
            step=100,
            key="map_cases_slider"
        )
        
    # Obtener el subconjunto de homicidios acumulados hasta la selección
    df_cumulative = df_homicidios_sorted.iloc[:map_cases_selected]
    
    # Determinar el rango de fechas actual
    if not df_cumulative.empty:
        first_date = df_cumulative['Fecha Infracción'].iloc[0]
        last_date = df_cumulative['Fecha Infracción'].iloc[-1]
        st.info(f"📅 **Período en el Mapa:** {first_date} hasta {last_date} ({map_cases_selected} de {total_casos_base} homicidios acumulados)")
        
    # Crear copia local de subcircuitos para el filtro dinámico
    gdf_map = gdf_subcircuitos.copy()
    
    # 1. Agrupar homicidios en este subconjunto por subcircuito
    conteo_sub_cum = df_cumulative.groupby(config.COL_COD_SUBCIRCUITO).size().to_dict()
    gdf_map['homicidios_totales'] = gdf_map['cod_subcircu'].map(conteo_sub_cum).fillna(0)
    
    # 2. Utilizar las estimaciones de Riesgo Relativo y Probabilidad de Excedencia reales del modelo bayesiano
    if 'rr_estimado_mediana' in gdf_subcircuitos.columns:
        gdf_map['rr_estimado_mediana'] = gdf_subcircuitos['rr_estimado_mediana']
        gdf_map['probabilidad_excedencia'] = gdf_subcircuitos['probabilidad_excedencia']
        gdf_map['rr_limite_inf_95'] = gdf_subcircuitos.get('rr_limite_inf_95', gdf_map['rr_estimado_mediana'])
        gdf_map['rr_limite_sup_95'] = gdf_subcircuitos.get('rr_limite_sup_95', gdf_map['rr_estimado_mediana'])
    else:
        # Si aún no se ha ejecutado el modelo bayesiano, indicar NaN
        gdf_map['rr_estimado_mediana'] = np.nan
        gdf_map['probabilidad_excedencia'] = np.nan
        gdf_map['rr_limite_inf_95'] = np.nan
        gdf_map['rr_limite_sup_95'] = np.nan
    
    col_m1, col_m2 = st.columns([3, 1.2])
    
    with col_m1:
        # Inicializar mapa centrado exactamente en el área urbana de Guayaquil
        m = folium.Map(location=[-2.18, -79.90], zoom_start=12.5, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", attr="Tiles © Esri")
        
        # Función para asignar color según riesgo
        def get_color(row):
            if opcion_mapa == "Probabilidad de Excedencia":
                val = row['probabilidad_excedencia']
                return '#ff4444' if val > 0.75 else '#ffb74d' if val > 0.40 else '#66bb6a'
            elif opcion_mapa == "Riesgo Relativo Estimado":
                val = row['rr_estimado_mediana']
                return '#ff4444' if val > 1.5 else '#ffb74d' if val > 0.9 else '#66bb6a'
            else:
                val = row['homicidios_totales']
                return '#ff4444' if val > 20 else '#ffb74d' if val > 5 else '#66bb6a'
                
        # Graficar círculos proporcionales sobre los centroides de los subcircuitos del set mapeado
        for idx, row in gdf_map.iterrows():
            if row.geometry is None:
                continue
            
            centroid = row.geometry.centroid
            c = get_color(row)
            
            # Calcular radio proporcional al riesgo relativo
            val_rr = row['rr_estimado_mediana']
            radius_m = int(max(200, min(1200, val_rr * 250)))
            
            # Crear popup html con diseño limpio
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 11px;">
                <b style="font-size: 12px; color: #58a6ff;">{row['subcircuito_nom']}</b><br>
                <b>Circuito:</b> {row['circuito_nom']}<br>
                <b>Distrito:</b> {row['distrito_nom']}<br>
                <hr style="margin: 4px 0; border: 0; border-top: 1px solid #30363d;">
                <b>Homicidios Reales:</b> {int(row['homicidios_totales'])}<br>
                <b>Riesgo Relativo (RR):</b> {row['rr_estimado_mediana']:.2f}x<br>
                <b>IC 95%:</b> [{row['rr_limite_inf_95']:.2f}, {row['rr_limite_sup_95']:.2f}]<br>
                <b>P(RR > 1.5):</b> <span style="color: {c}; font-weight: bold;">{row['probabilidad_excedencia']*100:.1f}%</span>
            </div>
            """
            
            folium.Circle(
                location=[centroid.y, centroid.x],
                radius=radius_m,
                color=c,
                fill=True,
                fillColor=c,
                fillOpacity=0.55,
                weight=1,
                tooltip=f"{row['subcircuito_nom']} (RR: {row['rr_estimado_mediana']:.2f}x)"
            ).add_child(folium.Popup(popup_html, max_width=250)).add_to(m)
            
        # Renderizar en Streamlit de forma fluida (forzando re-render mediante key dinámica)
        st_folium(m, width=900, height=580, key=f"main_map_{map_cases_selected}_{opcion_mapa}", returned_objects=[])

    with col_m2:
        # Calcular estadísticas del riesgo espacial de forma dinámica según el slider
        if total_homicidios > 0:
            max_rr = float(gdf_map['rr_estimado_mediana'].max())
            mean_rr = float(gdf_map['rr_estimado_mediana'].mean())
            min_rr = float(gdf_map['rr_estimado_mediana'].min())
            sub_greater_1 = int((gdf_map['rr_estimado_mediana'] > 1.0).sum())
            sub_less_1 = int((gdf_map['rr_estimado_mediana'] < 1.0).sum())
            
            # Ordenar para obtener Top 3 zonas de mayor riesgo
            top_3_zones = gdf_map.sort_values(by='rr_estimado_mediana', ascending=False).head(3)
            top_3_html = ""
            for i, (idx_z, row_z) in enumerate(top_3_zones.iterrows()):
                top_3_html += f"""<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; border-bottom: 1px solid #30363d; padding-bottom: 4px; margin-bottom: 4px;">
<span><b>{i+1}. {row_z['subcircuito_nom']}</b></span>
<span style="background-color: rgba(255,68,68,0.15); color: #ff4444; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold;">
RR: {row_z['rr_estimado_mediana']:.2f}x
</span>
</div>"""
                
            # Calcular la incertidumbre promedio
            gdf_map['incertidumbre_ancho'] = gdf_map['rr_limite_sup_95'] - gdf_map['rr_limite_inf_95']
            mean_unc_half = float(gdf_map['incertidumbre_ancho'].mean() / 2.0)
            
            # Buscar el subcircuito con mayor precisión (el intervalo más estrecho)
            most_precise = gdf_map.sort_values(by='incertidumbre_ancho', ascending=True).iloc[0]
            most_precise_name = most_precise['subcircuito_nom']
            most_precise_ci = f"[{most_precise['rr_limite_inf_95']:.2f}, {most_precise['rr_limite_sup_95']:.2f}]"
        else:
            max_rr, mean_rr, min_rr, sub_greater_1, sub_less_1 = 0, 0, 0, 0, 0
            top_3_html = "<div>Sin datos</div>"
            mean_unc_half = 0
            most_precise_name = "N/A"
            most_precise_ci = "[0, 0]"

        # Leer criterios de información dinámicos para los indicadores
        crit_path_side = os.path.join(config.TABLES_DIR, "tabla_criterios_informacion.xlsx")
        crit_waic_str = "N/D"
        crit_loo_str = "N/D"
        if os.path.exists(crit_path_side):
            try:
                df_c_side = pd.read_excel(crit_path_side)
                c_map_side = dict(zip(df_c_side['Métrica'], df_c_side['Valor']))
                if 'ELPD_WAIC' in c_map_side:
                    crit_waic_str = f"{c_map_side['ELPD_WAIC']:.1f}"
                if 'ELPD_LOO' in c_map_side:
                    crit_loo_str = f"{c_map_side['ELPD_LOO']:.1f}"
            except Exception:
                pass
        draws_cnt_str = f"{trace.posterior.dims.get('chain', 4) * trace.posterior.dims.get('draw', 1000):,}" if trace is not None else "4,000"

        # Panel derecho con estadísticas avanzadas y leyendas
        st.markdown(f"""<div class="stat-card" style="border-left: 5px solid #ff4444; margin-bottom: 12px; padding: 15px;">
<div style="font-size: 0.85rem; font-weight: bold; color: #ff4444; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
<i class="fa-solid fa-ranking-star"></i> 🏆 TOP ZONAS DE MAYOR RIESGO
</div>
<div style="display: flex; flex-direction: column; gap: 4px;">
{top_3_html}
</div>
</div>

<div class="stat-card" style="border-left: 5px solid #58a6ff; margin-bottom: 12px; padding: 15px;">
<div style="font-size: 0.85rem; font-weight: bold; color: #58a6ff; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
<i class="fa-solid fa-chart-bar"></i> 📊 RIESGO ESPACIAL (INDICADORES)
</div>
<div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.8rem;">
<div style="display: flex; justify-content: space-between;">
<span>RR máximo</span>
<strong style="color: #c9d1d9;">{max_rr:.2f}</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>RR promedio</span>
<strong style="color: #c9d1d9;">{mean_rr:.2f}</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>RR mínimo</span>
<strong style="color: #c9d1d9;">{min_rr:.2f}</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>Subcircuitos con RR > 1</span>
<strong style="color: #ffb74d;">{sub_greater_1}</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>Subcircuitos con RR < 1</span>
<strong style="color: #66bb6a;">{sub_less_1}</strong>
</div>
</div>
</div>

<div class="stat-card" style="border-left: 5px solid #ffb74d; margin-bottom: 12px; padding: 15px;">
<div style="font-size: 0.85rem; font-weight: bold; color: #ffb74d; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
<i class="fa-solid fa-microscope"></i> ⚖️ MODELO BAYESIANO (INDICADORES)
</div>
<div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.8rem;">
<div style="display: flex; justify-content: space-between;">
<span>ELPD-WAIC</span>
<strong style="color: #c9d1d9;">{crit_waic_str}</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>ELPD-LOO</span>
<strong style="color: #c9d1d9;">{crit_loo_str}</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>Escala</span>
<strong style="color: #66bb6a;">Log (ELPD)</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>Muestras Posteriores</span>
<strong style="color: #58a6ff;">{draws_cnt_str}</strong>
</div>
</div>
</div>

<div class="stat-card" style="border-left: 5px solid #66bb6a; margin-bottom: 12px; padding: 15px;">
<div style="font-size: 0.85rem; font-weight: bold; color: #66bb6a; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
<i class="fa-solid fa-circle-question"></i> 🛡️ INCERTIDUMBRE DEL ESTIMADOR (IC 95%)
</div>
<div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.8rem;">
<div style="display: flex; justify-content: space-between;">
<span>Precisión Promedio (±)</span>
<strong style="color: #c9d1d9;">{mean_unc_half:.2f}</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>Zona de Mayor Precisión</span>
<span style="font-size: 0.75rem; color: #8b949e; text-align: right;"><b>{most_precise_name}</b> <br>IC: {most_precise_ci}</span>
</div>
<div style="font-size: 0.7rem; color: #8b949e; margin-top: 6px; line-height: 1.3; text-align: justify; border-top: 1px solid #30363d; padding-top: 6px;">
El encogimiento bayesiano (shrinkage) del estimador espacial reduce artificialmente la incertidumbre en zonas con poblaciones pequeñas o bajo reporte de crímenes, evitando falsos positivos.
</div>
</div>
</div>""", unsafe_allow_html=True)

# ================= TAB 3: EVALUACIÓN BAYESIANA =================
with tab_bayes:
    st.subheader("Evaluación de Inferencia Bayesiana Variacional (ADVI)")
    st.markdown("Esta sección resume la optimización y bondad de ajuste del modelo Bayesiano jerárquico estimado mediante Inferencia Variacional por Diferenciación Automática (ADVI) en PyMC.")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**Métricas de Optimización Variacional (ADVI)**")
        
        # Leer diagnósticos de convergencia reales
        diag_path = os.path.join(config.TABLES_DIR, "tabla_diagnosticos_bayesianos.xlsx")
        draws_cnt_val = "2,000"
        crit_tol_val = "Variación 5k < 0.1%"
        k_gt_07_val = "N/D"
        k_adv_val = "N/D"
        model_eval_val = "Modelo (b) Binomial Negativa (ICAR) [Mejor LOO-CV]"
        if os.path.exists(diag_path):
            try:
                df_diag_conv = pd.read_excel(diag_path, sheet_name="Diagnosticos_Convergencia")
                diag_map = dict(zip(df_diag_conv['Diagnóstico'], df_diag_conv['Valor']))
                draws_cnt_val = f"{diag_map.get('Muestras Posteriores Extraídas', 2000):,}"
                crit_tol_val = str(diag_map.get('Criterio de Convergencia', 'Variación 5k < 0.1%'))
                k_gt_07_val = f"{diag_map.get('Pareto-k > 0.7 (Observaciones)', 'N/D')}"
                k_adv_val = f"{diag_map.get('Advertencia Pareto-k', 'N/D')}"
                model_eval_val = str(diag_map.get('Modelo Evaluado', model_eval_val))
            except Exception:
                pass
                
        if k_gt_07_val == "N/D" or str(k_gt_07_val).strip() == "N/D":
            pareto_k_display = "N/D"
        else:
            pareto_k_display = f"{k_gt_07_val} de 488 observaciones (5.5 %). El ELPD-LOO debe leerse con cautela."
                
        st.markdown(f"""
        - **Modelo de Referencia Evaluado:** **{model_eval_val}**
        - **Método de Inferencia:** ADVI (Automatic Differentiation Variational Inference, Adam lr=0.005).
        - **Limitación de ADVI:** subestima la desviación estándar posterior entre 38 % y 70 % respecto a NUTS (salvo en el parámetro de dispersión); por eso las probabilidades se validan con NUTS.
        - **Muestras Posteriores Extraídas:** {draws_cnt_val} draws de la aproximación variacional.
        - **Criterio de Convergencia:** {crit_tol_val}.
        - **ELPD-WAIC:** **{crit_waic_str}** (Escala logarítmica).
        - **ELPD-LOO:** **{crit_loo_str}** (Escala logarítmica).
        - **Criterio principal:** ELPD-LOO. El WAIC se reporta como complemento.
        - **Pareto-k > 0.7:** {pareto_k_display}
        - **Validación Metodológica:** El ajuste complementario con NUTS (MCMC) está en la pestaña «Modelo Bayesiano», subpestaña «Validación ADVI vs. NUTS».
        """)
            
    with col_d2:
        # Mostrar imagen del posterior predictivo PPC
        ppc_img_path = os.path.join(config.FIGURES_DIR, "Figura19_bayesian_ppc.png")
        if os.path.exists(ppc_img_path):
            st.image(ppc_img_path, caption="Comprobación predictiva posterior (PPC) en rango completo (ADVI)", width="stretch")
            st.caption("Las réplicas posteriores reproducen la forma asimétrica con cola larga de los conteos observados.")
        else:
            st.info("💡 La imagen de verificación predictiva posterior (PPC) se generará al ejecutar `python run_pipeline.py`.")

    # Mostrar imagen de convergencia del ELBO si existe
    elbo_img_path = os.path.join(config.FIGURES_DIR, "Figura_convergencia_elbo.png")
    if os.path.exists(elbo_img_path):
        st.image(elbo_img_path, caption="Trayectoria de optimización del ELBO hacia la meseta de convergencia", width="stretch")
    else:
        trace_img_path = os.path.join(config.FIGURES_DIR, "Figura14_bayesian_traceplots.png")
        if os.path.exists(trace_img_path):
            st.image(trace_img_path, caption="Traceplots de los parámetros posteriores de ADVI", width="stretch")
        else:
            st.info("💡 Ejecuta `python run_pipeline.py` en la consola para generar los gráficos de diagnóstico.")

# ================= TAB 3.5: MODELO BAYESIANO =================
with tab_modelo_bayesiano:
    st.header("🧠 Inferencia y Modelado Bayesiano Jerárquico")
    st.markdown("Esta sección presenta de manera detallada el modelo probabilístico espacio-temporal estimado mediante PyMC y ArviZ.")
    
    # Intentar cargar la traza
    if trace is None:
        st.warning("⚠️ No se encontró la traza de inferencia bayesiana `pymc_trace.nc`. Por favor ejecuta `run_pipeline.py` para entrenar el modelo.")
    else:
        
        # Sub-pestañas dentro de la sección bayesiana
        subtab_espec, subtab_post, subtab_comp, subtab_nuts, subtab_mapas, subtab_resumen = st.tabs([
            "📖 Pestaña 1: Especificación del Modelo",
            "📈 Pestaña 2: Distribuciones Posteriores",
            "📊 Pestaña 3: Comparación del Modelo",
            "⚖️ Pestaña 4: Validación ADVI vs. NUTS",
            "🗺️ Pestaña 5: Mapas Bayesianos",
            "📝 Pestaña 6: Resumen del Modelo"
        ])
        
        # Obtener variables e índices para mapeo
        if df_anual_map is not None:
            sub_codes_loc = sorted(df_anual_map['cod_subcircu'].unique())
            anios_loc = sorted(df_anual_map['Anio'].unique())
        else:
            sub_codes_loc = []
            anios_loc = []
            
        # Pestaña 1: Especificación del Modelo
        with subtab_espec:
            st.subheader("Especificación Matemática y Estructura Jerárquica")
            st.markdown("""
            El modelo ajustado para estimar los conteos esperados y el riesgo relativo de muertes violentas en Guayaquil se formula como un modelo Bayesiano Jerárquico Espacio-Temporal con verosimilitud **Binomial Negativa** para controlar la sobredispersión territorial extrema.
            
            #### 1. Modelo de referencia seleccionado
            **Modelo (b): Binomial Negativa con ICAR**
            
            - **Verosimilitud (Likelihood):**
              $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
            
            - **Función de Enlace (Log-Link):**
              $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i$$
            
            - **Componente Espacial Estructurado (ICAR):**
              $$\\phi \\sim \\text{ICAR}(W, \\sigma_\\phi^2)$$
              
            Donde:
            - $Y_{it}$: Número de homicidios observados en el subcircuito $i$ en el año $t$.
            - $\\mu_{it}$: Media esperada condicional ajustada por el modelo.
            - $\\alpha$: Intercepto global: logaritmo del conteo esperado en 2024 para un subcircuito con log(Área) = 0 y efecto espacial nulo. No es una tasa (el modelo no usa población).
            - $\\beta_{\\text{year}}$: Efecto fijo de año 2025 frente a la línea base 2024 con prior $\\text{Normal}(0, 1)$.
            - $\\beta_{\\text{area}}$: Coeficiente para $\\log(\\text{Area}_i)$ con prior $\\text{Normal}(0, 1)$.
            - $\\sigma_\\phi$: Escala del efecto espacial intrínseco autorregresivo condicional con prior $\\text{Exponential}(1.0)$.
            - $\\alpha_{disp}$: Parámetro de dispersión de la verosimilitud Binomial Negativa con prior $\\text{Exponential}(1.0)$.
            
            *Justificación Metodológica:* El Modelo (b) obtiene el mayor ELPD-LOO = −1,233.75 y la menor cantidad de observaciones con Pareto-$k > 0.7$ (27 observaciones frente a 38 en el Modelo (c) y 35 en el Modelo (d)). La diferencia frente al Modelo (c) (−1,241.65) es de 7.90 puntos con dse = 4.18 (|$ \\Delta $| < 2·dse = 8.36), lo que no permite distinguirlos con claridad. Por parsimonia (menor número de parámetros al no requerir $\\sigma_\\theta$ ni efectos no estructurados $\\theta_i$), se selecciona el Modelo (b) como el modelo de referencia final de la tesis.
            
            *Alcance del modelo:* no incluye offset poblacional (no hay datos de población por subcircuito) ni interacción espacio-tiempo; el efecto de año es común a todos los subcircuitos. Se modelan 2024 y 2025 (488 observaciones en 244 subcircuitos); enero de 2026 queda fuera.
            
            #### 2. Modelos Estructurales Comparativos
            - **Poisson jerárquico (referencia):** misma estructura lineal que el modelo (b), con efecto espacial ICAR y verosimilitud Poisson (sin parámetro de dispersión).
            - **Modelo (c): Besag ICAR + Efecto No Estructurado (Besag et al., 1991):**
              $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i + \\theta_i$$
              donde $\\theta_i \\sim \\text{Normal}(0, \\sigma_\\theta^2)$ con $\\sigma_\\theta \\sim \\text{Exponential}(1.0)$.
            - **Modelo (d): BYM2 Binomial Negativo (Riebler et al., 2016):**
              $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\psi_i$$

              $$\\psi_i = \\sigma_s \\left( \\sqrt{\\rho} \\phi^*_i + \\sqrt{1 - \\rho} \\theta^*_i \\right)$$

              donde $\\rho \\sim \\text{Beta}(0.5, 0.5)$ es el parámetro de mezcla espacial.
            
            #### 3. Distribuciones a Priori (Priors)
            Convención: Normal(media, desviación estándar); Exponential(tasa).
            
            - **Intercepto Global:** $\\alpha \\sim \\text{Normal}(0, 2)$
            - **Contraste Temporal Anual:** $\\beta_{\\text{year}} \\sim \\text{Normal}(0, 1)$
            - **Elasticidad del Área:** $\\beta_{\\text{area}} \\sim \\text{Normal}(0, 1)$
            - **Escala Estructurada ICAR:** $\\sigma_\\phi \\sim \\text{Exponential}(1.0)$
            - **Parámetro de Dispersión:** $\\alpha_{disp} \\sim \\text{Exponential}(1.0)$
            - **Escala No Estructurada (Modelo c):** $\\sigma_\\theta \\sim \\text{Exponential}(1.0)$
            - **Escala Total y Mezcla (Modelo d):** $\\sigma_s \\sim \\text{Exponential}(1.0)$, $\\rho \\sim \\text{Beta}(0.5, 0.5)$
            """)
            
        # Pestaña 2: Distribuciones Posteriores
        with subtab_post:
            st.subheader("Distribuciones Posteriores de los Parámetros Globales")
            st.markdown("Densidades marginales posteriores e intervalos HDI al 94 % de los parámetros globales del modelo de referencia (b), estimados con ADVI.")
            
            st.info("Estas densidades provienen de ADVI, que subestima la desviación estándar posterior entre 38 % y 70 % respecto a NUTS. Los intervalos reales son más anchos (por ejemplo, para beta_year: 0.25–0.39 en ADVI frente a 0.20–0.42 en NUTS). La comparación completa está en la subpestaña «Validación ADVI vs. NUTS».")
            post_vars = [v for v in ['alpha', 'beta_year', 'beta_area', 'sigma_phi', 'sigma_theta', 'sigma_s', 'rho', 'alpha_disp'] if v in trace.posterior]
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            az.plot_posterior(trace, var_names=post_vars, ax=axes, round_to=3)
            for ax in axes.flat[len(post_vars):]:
                ax.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Intervalos HDI al 94 % (valor por defecto de ArviZ). Aproximación variacional ADVI.")
            
            rho_advi_str, rho_nuts_str = get_rho_advi_nuts_str()
            b_year_exp = float(np.exp(trace.posterior['beta_year'].mean().item())) if 'beta_year' in trace.posterior else 1.38
            b_area_mean = float(trace.posterior['beta_area'].mean().item()) if 'beta_area' in trace.posterior else 0.32
            
            st.markdown(f"""
            *Interpretación:*
            - **alpha:** logaritmo del conteo esperado en 2024 para un subcircuito de área unitaria (log(Área) = 0) y efecto espacial nulo. No es una tasa.
            - **beta_year:** efecto de 2025 frente a 2024 en escala logarítmica. exp(beta_year) ≈ {b_year_exp:.2f} es el riesgo relativo de 2025 respecto a 2024.
            - **beta_area:** elasticidad del área: un aumento de 1 % en el área del subcircuito se asocia con aproximadamente un {b_area_mean:.2f} % más de homicidios esperados.
            - **sigma_phi:** escala del efecto espacial estructurado ICAR (modelo de referencia b).
            - **alpha_disp:** parámetro de dispersión de la Binomial Negativa; Var = μ + μ²/alpha_disp, por lo que un valor mayor implica menos sobredispersión.
            """)
            
            with st.expander("Parámetros de los modelos comparativos (c) y (d)"):
                st.markdown(f"""
                - **sigma_theta:** Escala de la heterogeneidad zonal no estructurada (Modelo c).
                - **sigma_s:** Desviación estándar total del componente espacial BYM2 (Modelo comparativo d).
                - **rho:** Parámetro de mezcla espacial con prior Beta(0.5, 0.5) en el modelo BYM2. *Aviso metodológico:* en ADVI se estima en ~{rho_advi_str} y en NUTS en ~{rho_nuts_str}; por tanto, su estimación bajo ADVI no es fiable para cuantificar la proporción de varianza espacial por contigüidad.
                """)
                    
        # Pestaña 3: Comparación del Modelo
        with subtab_comp:
            st.subheader("Evaluación de Información Predictiva y Ajuste Out-of-Sample")
            
            comp_tab1, comp_tab2 = st.tabs([
                "📊 Sobredispersión y Criterios de Información",
                "🧠 Comparación entre Modelos Bayesianos"
            ])
            
            with comp_tab1:
                st.markdown("Evidencia de sobredispersión con modelos clásicos de máxima verosimilitud (Poisson y Binomial Negativa, con las mismas covariables: año y log del área, sin efecto espacial) y criterios de información del modelo bayesiano de referencia (b). El AIC/BIC y el ELPD no están en la misma escala y no se comparan entre sí.")
                
                # Cargar métricas reales desde outputs/tables/
                excel_crit_path = os.path.join(config.TABLES_DIR, "tabla_criterios_informacion.xlsx")
                excel_clas_path = os.path.join(config.TABLES_DIR, "tabla_modelos_clasicos.xlsx")
                
                waic_val = "N/D"
                loo_val = "N/D"
                if os.path.exists(excel_crit_path):
                    try:
                        df_crit = pd.read_excel(excel_crit_path)
                        crit_map = dict(zip(df_crit['Métrica'], df_crit['Valor']))
                        if 'ELPD_WAIC' in crit_map:
                            waic_val = f"{crit_map['ELPD_WAIC']:,.2f}"
                        elif 'WAIC' in crit_map:
                            waic_val = f"{crit_map['WAIC']:,.2f}"
                        if 'ELPD_LOO' in crit_map:
                            loo_val = f"{crit_map['ELPD_LOO']:,.2f}"
                        elif 'LOO' in crit_map:
                            loo_val = f"{crit_map['LOO']:,.2f}"
                    except Exception as e:
                        st.error(f"Error al leer criterios de información: {e}")
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric(label="ELPD-WAIC, modelo (b) (escala log)", value=waic_val)
                with col_c2:
                    st.metric(label="ELPD-LOO, modelo (b) (escala log)", value=loo_val)
                
                if os.path.exists(excel_clas_path):
                    st.markdown("#### Comparación Frecuentista Clásica (Evidencia de Sobredispersión):")
                    try:
                        df_clas = pd.read_excel(excel_clas_path)
                        st.dataframe(df_clas.style.format(precision=2), hide_index=True, width="stretch")
                    except Exception as e:
                        st.error(f"Error al leer tabla clásica: {e}")
                
                st.markdown(f"""
                #### Evaluación del Desempeño Predictivo y Generalización:
                El modelo bayesiano jerárquico ajustado presenta un **ELPD-WAIC** de **{waic_val}** y un **ELPD-LOO** de **{loo_val}** (en escala logarítmica de densidad predictiva esperada).
                
                1. **Capacidad predictiva (WAIC / LOO-CV):** ambos criterios estiman la densidad predictiva esperada para datos no observados penalizando la complejidad efectiva. Un ELPD mayor indica mejor capacidad predictiva esperada; el modelo (b) tiene el mayor ELPD-LOO entre los modelos comparados. Con 27 observaciones con Pareto-k > 0.7, el ELPD-LOO debe leerse con cautela.
                2. **Tratamiento de la sobredispersión:** la verosimilitud **Binomial Negativa** introduce el parámetro $\\alpha_{{disp}}$ y permite que la varianza supere a la media (Var = μ + μ²/α_disp). En los datos, la razón varianza/media es 29.03, y el AIC baja de 7107.31 (Poisson) a 3114.81 (Binomial Negativa).
                3. **Suavizado espacial:** el componente espacial **ICAR** comparte información entre subcircuitos vecinos y estabiliza las estimaciones en zonas con pocos registros. El efecto de año es un efecto fijo y no se suaviza.
                """)
                
            with comp_tab2:
                st.markdown("Comparación de bondad de ajuste y capacidad predictiva entre cuatro formulaciones bayesianas jerárquicas, ajustadas con ADVI (Adam, lr = 0.005, 50,000 iteraciones).")
                
                # Cargar tabla comparativa bayesiana real generada por stage8
                excel_comp_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_modelos_bayesiana.xlsx")
                if os.path.exists(excel_comp_path):
                    try:
                        df_comp_bayes = pd.read_excel(excel_comp_path)
                        cols_bayes = [
                            'Modelo', 'elpd_loo', 'se', 'elpd_diff', 'dse',
                            'weight', 'p_loo', 'k > 0.7', 'elpd_waic'
                        ]
                        rename_bayes = {
                            'Modelo': 'Modelo',
                            'elpd_loo': 'ELPD-LOO',
                            'se': 'SE',
                            'elpd_diff': 'Δ ELPD',
                            'dse': 'dse',
                            'weight': 'Peso (stacking)',
                            'p_loo': 'p_loo',
                            'k > 0.7': 'Pareto-k > 0.7',
                            'elpd_waic': 'ELPD-WAIC'
                        }
                        df_mostrar = df_comp_bayes[cols_bayes].rename(columns=rename_bayes)
                        st.dataframe(
                            df_mostrar,
                            hide_index=True,
                            width="stretch",
                            column_config={
                                "ELPD-LOO": st.column_config.NumberColumn(format="%.2f"),
                                "SE": st.column_config.NumberColumn(format="%.2f"),
                                "Δ ELPD": st.column_config.NumberColumn(format="%.2f"),
                                "dse": st.column_config.NumberColumn(format="%.2f"),
                                "Peso (stacking)": st.column_config.NumberColumn(format="%.3f"),
                                "p_loo": st.column_config.NumberColumn(format="%.2f"),
                                "Pareto-k > 0.7": st.column_config.NumberColumn(format="%d"),
                                "ELPD-WAIC": st.column_config.NumberColumn(format="%.2f"),
                            }
                        )
                        
                        st.info("📌 Nota metodológica sobre selección de modelos: el modelo (b) Binomial Negativa (ICAR) tiene el mayor ELPD-LOO (−1,233.75; SE = 26.76) y el menor número de observaciones con Pareto-k > 0.7 (27, frente a 38 en el modelo (c), 35 en el (d) y 92 en el Poisson jerárquico). Frente al modelo (c), la diferencia es de 7.90 puntos con dse = 4.18, menor que 2·dse, por lo que no se distinguen con claridad. Frente al modelo (d), la diferencia es de 9.17 con dse = 4.35, mayor que 2·dse. Frente al Poisson es de 135.47. El peso de apilamiento (stacking) de (b) es 0.99. Por parsimonia (menos parámetros) y mejor diagnóstico de Pareto-k, se adopta el modelo (b) como referencia de la tesis.")
                        
                        st.caption("Las formulaciones matemáticas de los modelos comparados están en la Pestaña 1: Especificación del Modelo.")
                    except Exception as e:
                        st.error(f"Error al leer la tabla de comparación bayesiana: {e}")
                else:
                    st.warning("⚠️ La tabla de comparación de modelos (`tabla_comparacion_modelos_bayesiana.xlsx`) aún no ha sido generada. Ejecuta `python run_pipeline.py` en la terminal para ajustar formalmente los 4 modelos mediante ADVI.")
            
        # Pestaña 4: Validación ADVI vs. NUTS
        with subtab_nuts:
            st.subheader("Validación Metodológica: ADVI vs. NUTS (MCMC) y Estabilidad de Semillas")
            st.markdown("""
            **Justificación Metodológica:**
            - **ADVI (Automatic Differentiation Variational Inference)** es el método de inferencia principal de la tesis por su alta eficiencia computacional en modelos espaciales jerárquicos de alta dimensión ($V=244$ nodos, múltiples hiperparámetros de escala y sobredispersión).
            - **NUTS (No-U-Turn Sampler, MCMC)** se utiliza estrictamente como **patrón de validación metodológica** sobre el modelo de referencia ganador (Modelo b: Binomial Negativa ICAR), así como sobre los Modelos (c) y (d), empleando exactamente la misma formulación y los mismos priors.
            - Esta validación indica que ADVI conserva el ordenamiento de los riesgos relativos espaciales (correlaciones de Pearson y de Spearman superiores a 0.99 en los tres modelos) y reproduce los coeficientes fijos del modelo (b): $\\beta_{\\text{year}}$ = 0.321 en ADVI frente a 0.310 en NUTS y $\\beta_{\\text{area}}$ = 0.324 frente a 0.322.
            - **Incertidumbre y probabilidades de excedencia:** ADVI subestima la desviación estándar posterior entre 38 % y 70 % respecto a NUTS (salvo en $\\alpha_{\\text{disp}}$), por lo que las probabilidades $P(RR > 1.5)$ bajo ADVI se saturan: 37 subcircuitos tienen $P \\geq 0.90$ con ADVI frente a 30 con NUTS. Ejemplo: Nuevo Guayaquil 3 tiene RR mediano 4.39 con IC 95 % [2.79, 6.91] en ADVI, y 4.20 con [2.30, 8.14] en NUTS. Por eso las probabilidades que se reportan en la tesis son las de NUTS para el modelo final; con 2 cadenas, también son una aproximación.
            """)
            
            nuts_tab1, nuts_tab2 = st.tabs([
                "⚖️ Comparación Paramétrica ADVI vs. NUTS",
                "🌱 Estabilidad entre Semillas (ADVI)"
            ])
            
            with nuts_tab1:
                st.markdown("#### Tabla Comparativa de Parámetros: ADVI vs. NUTS (Modelos b, c y d)")
                nuts_comp_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_advi_nuts.xlsx")
                if os.path.exists(nuts_comp_path):
                    try:
                        df_nuts_comp = pd.read_excel(nuts_comp_path)
                        st.dataframe(df_nuts_comp.style.format(precision=3), hide_index=True, width="stretch", height=580)
                        
                        rho_advi_val, rho_nuts_val = get_rho_advi_nuts_str()
                        st.warning(f"""
                        ⚠️ **Aviso Metodológico Riguroso sobre Hiperparámetros:**
                        - **Coeficientes fijos (modelo b):** ADVI reproduce el contraste de año ($\\beta_{{\\text{{year}}}}$: 0.321 vs. 0.310) y la elasticidad del área ($\\beta_{{\\text{{area}}}}$: 0.324 vs. 0.322). En los modelos (c) y (d), $\\beta_{{\\text{{area}}}}$ difiere más (≈ 0.32 en ADVI frente a ≈ 0.23 en NUTS).
                        - **Subestimación de incertidumbre:** la desviación estándar de ADVI es entre 38 % y 70 % menor que la de NUTS en $\\alpha$, $\\beta_{{\\text{{year}}}}$, $\\beta_{{\\text{{area}}}}$ y $\\sigma_\\phi$; en $\\alpha_{{\\text{{disp}}}}$ no hay subestimación (cociente 1.01).
                        - **Hiperparámetros:** en el modelo (b), $\\sigma_\\phi$ es 2.59 con ADVI y 2.05 con NUTS, y $\\alpha_{{\\text{{disp}}}}$ es 6.98 frente a 5.12. En el BYM2, el parámetro de mezcla $\\rho$ se estima en ~{rho_advi_val} con ADVI y en ~{rho_nuts_val} con NUTS, aunque esa estimación de NUTS es poco precisa (R-hat 1.048, ESS_tail 41). Las estimaciones de hiperparámetros de ADVI deben presentarse junto a las de NUTS.
                        """)
                    except Exception as e:
                        st.error(f"Error al cargar la tabla de validación ADVI vs. NUTS: {e}")
                else:
                    st.info("La tabla comparativa ADVI vs. NUTS se generará al ejecutar `python run_pipeline.py`.")

                st.markdown("#### Diagnósticos de Muestreo NUTS (R-hat, Tamaño Efectivo y Divergencias)")
                diag_nuts_path = os.path.join(config.TABLES_DIR, "tabla_diagnosticos_nuts.xlsx")
                if os.path.exists(diag_nuts_path):
                    try:
                        df_diag_nuts = pd.read_excel(diag_nuts_path)
                        st.dataframe(df_diag_nuts, hide_index=True, width="stretch")
                        st.caption("Con 2 cadenas de 1,000 muestras, el modelo (b) alcanza R-hat máximo de 1.008 y ESS_bulk mínimo de 337 (umbral habitual: R-hat < 1.01 y ESS > 400). Los modelos (c) y (d) tienen R-hat de 1.013 y 1.048 y ESS_bulk de 119 y 68, por lo que su comparación con ADVI es orientativa.")
                    except Exception as e:
                        st.error(f"Error al cargar tabla de diagnósticos NUTS: {e}")
                    
                st.markdown("#### Correlación de Riesgos Relativos Espaciales (RR) por Subcircuito (ADVI vs. NUTS)")
                corr_path = os.path.join(config.TABLES_DIR, "tabla_correlacion_rr_advi_nuts.xlsx")
                if os.path.exists(corr_path):
                    try:
                        df_corr_rr = pd.read_excel(corr_path)
                        st.dataframe(
                            df_corr_rr.style.format(
                                {"Pearson_p_valor": "{:.1e}", "Spearman_p_valor": "{:.1e}"},
                                precision=4
                            ),
                            hide_index=True,
                            width="stretch"
                        )
                    except Exception as e:
                        st.error(f"Error al cargar tabla de correlaciones RR: {e}")
                        
                st.markdown("#### Dispersión y Ajuste de Riesgos Relativos (RR) por Subcircuito (244 Subcircuitos)")
                rr_scatter_img = os.path.join(config.FIGURES_DIR, "Figura_dispersion_rr_advi_nuts.png")
                if os.path.exists(rr_scatter_img):
                    st.image(rr_scatter_img, width="stretch")
                    st.caption("ADVI conserva el orden de los riesgos relativos (Pearson 0.998 y Spearman 0.998 en el modelo b), pero tiende a inflar los valores altos: la pendiente de la regresión ADVI–NUTS es 1.07 en el modelo (b).")
                else:
                    st.info("El gráfico de dispersión de RR ADVI vs. NUTS se generará al ejecutar `python run_pipeline.py`.")
                    
            with nuts_tab2:
                st.markdown("#### Evaluación de Estabilidad Estocástica entre Semillas Independientes (ADVI)")
                st.markdown("""
                Para evaluar si el resultado depende de la semilla, el modelo de referencia ganador (b) (Binomial Negativa ICAR) se evaluó bajo 3 semillas aleatorias independientes (42, 123 y 999) bajo el **criterio formal de convergencia unificado**: la variación de la media móvil del ELBO en las últimas 5,000 iteraciones debe ser menor al 0.1% (< 0.001) respecto de las 5,000 previas.
                """)
                seed_stab_path = os.path.join(config.TABLES_DIR, "tabla_estabilidad_semillas_advi.xlsx")
                if os.path.exists(seed_stab_path):
                    try:
                        df_seed_stab = pd.read_excel(seed_stab_path)
                        df_seed_stab['Semilla'] = df_seed_stab['Semilla'].astype(str)
                        st.dataframe(df_seed_stab.style.format(precision=4), hide_index=True, width="stretch")
                        st.warning("Las semillas 42 y 999 cumplen el criterio de convergencia; la semilla 123 no lo cumple (variación 0.20 %, frente al 0.1 % exigido). Las medias de los parámetros son estables entre semillas (desviación entre semillas de $\\beta_{\\text{year}}$ = 0.008); $\\alpha_{\\text{disp}}$ es el parámetro más sensible (6.98 a 7.55).")
                    except Exception as e:
                        st.error(f"Error al cargar la tabla de estabilidad de semillas: {e}")
                else:
                    st.info("La tabla de estabilidad de semillas se generará al ejecutar `python run_pipeline.py`.")
            
        # Pestaña 5: Mapas Bayesianos
        with subtab_mapas:
            # 1. Título y subtítulo
            st.markdown(r"""<div style="margin-bottom: 12px;">
<h2 style="color: #ffffff; font-weight: 700; margin: 0 0 4px 0; font-size: 24px;">Mapa probabilístico de riesgo</h2>
<div style="color: #8b949e; font-size: 13.5px;">Probabilidad posterior de que el riesgo relativo supere 1,5 · muestreo NUTS · 244 subcircuitos de Guayaquil</div>
</div>""", unsafe_allow_html=True)
            
            # 2. Selector de capa horizontal
            opciones_capa = [
                "P(RR>1,5) NUTS",
                "P(RR>1,0) NUTS",
                "RR mediano NUTS",
                "RR mediano ADVI",
                "Amplitud IC 95 %"
            ]
            
            # Estilos CSS específicos para botones del selector horizontal
            st.markdown(r"""<style>
div[data-testid="stSegmentedControl"] {
    margin-bottom: 12px;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background-color: #ff3d81 !important;
    color: #ffffff !important;
    border-color: #ff3d81 !important;
    font-weight: 700 !important;
}
div[data-testid="stSegmentedControl"] button {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 0px !important;
}
div[role="radiogroup"] label[data-checked="true"] {
    background-color: #ff3d81 !important;
    color: #ffffff !important;
}
</style>""", unsafe_allow_html=True)
            
            if hasattr(st, "segmented_control"):
                capa_sel = st.segmented_control(
                    "Selección de capa",
                    options=opciones_capa,
                    default="P(RR>1,5) NUTS",
                    label_visibility="collapsed",
                    key="mapa_tipo_bayes_seg"
                ) or "P(RR>1,5) NUTS"
            else:
                capa_sel = st.radio(
                    "Selección de capa",
                    options=opciones_capa,
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="mapa_tipo_bayes_rad"
                )

            # Cargar datos requeridos (solo lectura, sin recalcular)
            csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
            if not os.path.exists(csv_path):
                csv_path = "outputs/data/resultados_riesgo_relativo.csv"
                
            geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
            if not os.path.exists(geojson_path):
                geojson_path = "outputs/data/subcircuitos_riesgo_completo.geojson"
                
            if not os.path.exists(csv_path) or not os.path.exists(geojson_path):
                st.warning("⚠️ No se encontraron los archivos de datos (`resultados_riesgo_relativo.csv` o `subcircuitos_riesgo_completo.geojson`).")
            else:
                df_rr = pd.read_csv(csv_path)
                gdf_raw = gpd.read_file(geojson_path)
                
                # Unir por cod_subcircu
                base_gdf_cols = ['cod_subcircu', 'geometry']
                if 'subcircuito_nom' in gdf_raw.columns:
                    base_gdf_cols.append('subcircuito_nom')
                if 'distrito_nom' in gdf_raw.columns:
                    base_gdf_cols.append('distrito_nom')
                gdf_base = gdf_raw[base_gdf_cols].copy()
                
                rr_cols_merge = [c for c in df_rr.columns if c not in base_gdf_cols or c == 'cod_subcircu']
                gdf_map_bayes = gdf_base.merge(df_rr[rr_cols_merge], on='cod_subcircu', how='inner')
                
                if 'subcircuito_nom' not in gdf_map_bayes.columns and 'subcircuito_nom' in df_rr.columns:
                    gdf_map_bayes['subcircuito_nom'] = df_rr['subcircuito_nom']
                if 'distrito_nom' not in gdf_map_bayes.columns and 'distrito_nom' in df_rr.columns:
                    gdf_map_bayes['distrito_nom'] = df_rr['distrito_nom']
                    
                gdf_map_bayes['amplitud_ic_nuts'] = (
                    gdf_map_bayes['rr_limite_sup_95_nuts'] - gdf_map_bayes['rr_limite_inf_95_nuts']
                )
                
                # Paleta de 4 clases
                COLOR_CLASE = {
                    4: '#ff3d81',  # Clase 4 (más alta)
                    3: '#ff9e3d',  # Clase 3
                    2: '#7b5ea7',  # Clase 2
                    1: '#232b36'   # Clase 1 (más baja)
                }
                
                # Clasificación según la capa seleccionada
                if capa_sel == "P(RR>1,5) NUTS":
                    val_s = gdf_map_bayes['prob_exced_1_5_nuts']
                    def get_cls(v):
                        if v >= 0.90: return 4
                        elif v >= 0.50: return 3
                        elif v >= 0.10: return 2
                        else: return 1
                    gdf_map_bayes['_clase_num'] = val_s.map(get_cls)
                    clases_info = [
                        (4, "Muy alta", "≥ 0.90", int((gdf_map_bayes['_clase_num'] == 4).sum())),
                        (3, "Alta", "0.50 – 0.90", int((gdf_map_bayes['_clase_num'] == 3).sum())),
                        (2, "Moderada", "0.10 – 0.50", int((gdf_map_bayes['_clase_num'] == 2).sum())),
                        (1, "Baja", "< 0.10", int((gdf_map_bayes['_clase_num'] == 1).sum())),
                    ]
                    titulo_clases = "CLASES DE PROBABILIDAD"
                    subtitulo_a = "con P(RR > 1,5) ≥ 0,90 según NUTS"
                    
                elif capa_sel == "P(RR>1,0) NUTS":
                    val_s = gdf_map_bayes['prob_exced_1_0_nuts']
                    def get_cls(v):
                        if v >= 0.90: return 4
                        elif v >= 0.50: return 3
                        elif v >= 0.10: return 2
                        else: return 1
                    gdf_map_bayes['_clase_num'] = val_s.map(get_cls)
                    clases_info = [
                        (4, "Muy alta", "≥ 0.90", int((gdf_map_bayes['_clase_num'] == 4).sum())),
                        (3, "Alta", "0.50 – 0.90", int((gdf_map_bayes['_clase_num'] == 3).sum())),
                        (2, "Moderada", "0.10 – 0.50", int((gdf_map_bayes['_clase_num'] == 2).sum())),
                        (1, "Baja", "< 0.10", int((gdf_map_bayes['_clase_num'] == 1).sum())),
                    ]
                    titulo_clases = "CLASES DE PROBABILIDAD"
                    subtitulo_a = "con P(RR > 1,0) ≥ 0,90 según NUTS"
                    
                elif capa_sel == "RR mediano NUTS":
                    val_s = gdf_map_bayes['rr_mediana_nuts']
                    def get_cls(v):
                        if v >= 2.5: return 4
                        elif v >= 1.5: return 3
                        elif v >= 1.0: return 2
                        else: return 1
                    gdf_map_bayes['_clase_num'] = val_s.map(get_cls)
                    clases_info = [
                        (4, "Muy alto", "≥ 2.5", int((gdf_map_bayes['_clase_num'] == 4).sum())),
                        (3, "Alto", "1.5 – 2.5", int((gdf_map_bayes['_clase_num'] == 3).sum())),
                        (2, "Moderado", "1.0 – 1.5", int((gdf_map_bayes['_clase_num'] == 2).sum())),
                        (1, "Bajo", "< 1.0", int((gdf_map_bayes['_clase_num'] == 1).sum())),
                    ]
                    titulo_clases = "CLASES DE RIESGO RELATIVO"
                    subtitulo_a = "con RR mediano ≥ 2,5 según NUTS"
                    
                elif capa_sel == "RR mediano ADVI":
                    val_s = gdf_map_bayes['rr_estimado_mediana']
                    def get_cls(v):
                        if v >= 2.5: return 4
                        elif v >= 1.5: return 3
                        elif v >= 1.0: return 2
                        else: return 1
                    gdf_map_bayes['_clase_num'] = val_s.map(get_cls)
                    clases_info = [
                        (4, "Muy alto", "≥ 2.5", int((gdf_map_bayes['_clase_num'] == 4).sum())),
                        (3, "Alto", "1.5 – 2.5", int((gdf_map_bayes['_clase_num'] == 3).sum())),
                        (2, "Moderado", "1.0 – 1.5", int((gdf_map_bayes['_clase_num'] == 2).sum())),
                        (1, "Bajo", "< 1.0", int((gdf_map_bayes['_clase_num'] == 1).sum())),
                    ]
                    titulo_clases = "CLASES DE RIESGO RELATIVO (ADVI)"
                    subtitulo_a = "con RR mediano ≥ 2,5 según ADVI"
                    
                else: # Amplitud IC 95 %
                    val_s = gdf_map_bayes['amplitud_ic_nuts']
                    q_codes, q_bins = pd.qcut(val_s, q=4, labels=[1, 2, 3, 4], retbins=True)
                    gdf_map_bayes['_clase_num'] = q_codes.astype(int)
                    clases_info = [
                        (4, "Muy alta", f"({q_bins[3]:.2f}, {q_bins[4]:.2f}]", int((gdf_map_bayes['_clase_num'] == 4).sum())),
                        (3, "Alta", f"({q_bins[2]:.2f}, {q_bins[3]:.2f}]", int((gdf_map_bayes['_clase_num'] == 3).sum())),
                        (2, "Moderada", f"({q_bins[1]:.2f}, {q_bins[2]:.2f}]", int((gdf_map_bayes['_clase_num'] == 2).sum())),
                        (1, "Baja", f"[{q_bins[0]:.2f}, {q_bins[1]:.2f}]", int((gdf_map_bayes['_clase_num'] == 1).sum())),
                    ]
                    titulo_clases = "CUARTILES DE INCERTIDUMBRE"
                    subtitulo_a = "en cuartil superior (mayor amplitud)"
                    
                gdf_map_bayes['_fill_color'] = gdf_map_bayes['_clase_num'].map(COLOR_CLASE)
                
                # Preformatear datos para tooltip
                gdf_map_bayes['Subcircuito'] = gdf_map_bayes['subcircuito_nom'].astype(str).str.title()
                gdf_map_bayes['P(RR>1,5) NUTS'] = gdf_map_bayes['prob_exced_1_5_nuts'].map(lambda x: f"{x:.3f}")
                gdf_map_bayes['RR mediano (NUTS)'] = gdf_map_bayes.apply(
                    lambda r: f"{r['rr_mediana_nuts']:.2f} [{r['rr_limite_inf_95_nuts']:.2f}, {r['rr_limite_sup_95_nuts']:.2f}]", axis=1
                )
                gdf_map_bayes['RR mediano (ADVI)'] = gdf_map_bayes['rr_estimado_mediana'].map(lambda x: f"{x:.2f}")
                
                # Layout en dos columnas: [2.1, 1]
                col_map, col_panel = st.columns([2.1, 1])
                
                with col_map:
                    from folium.plugins import MiniMap

                    if st.button("Restablecer vista", key="btn_reset_mapa_prob"):
                        st.session_state['reset'] = st.session_state.get('reset', 0) + 1

                    # Mapa base oscura sin API key (Esri World Dark Gray Base) con límites estrictos
                    m_bayes = folium.Map(
                        location=[-2.17, -79.93],
                        zoom_start=12,
                        min_zoom=10,
                        max_zoom=16,
                        tiles=None,
                        control_scale=True
                    )
                    m_bayes.options["maxBounds"] = [[-2.45, -80.20], [-1.85, -79.60]]   # evita perderse en el mapa mundial
                    m_bayes.options["maxBoundsViscosity"] = 0.8

                    folium.TileLayer(
                        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                        attr="Esri",
                        name="Esri Dark Gray"
                    ).add_to(m_bayes)
                    
                    # Polígonos con GeoJson
                    folium.GeoJson(
                        gdf_map_bayes,
                        style_function=lambda feature: {
                            'fillColor': feature['properties']['_fill_color'],
                            'color': '#0d1117',
                            'weight': 0.5,
                            'fillOpacity': 0.9 if feature['properties']['_clase_num'] == 1 else 0.45
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['Subcircuito', 'P(RR>1,5) NUTS', 'RR mediano (NUTS)', 'RR mediano (ADVI)'],
                            aliases=['Subcircuito:', 'P(RR>1,5) NUTS:', 'RR mediano NUTS (IC 95 %):', 'RR mediano ADVI:'],
                            localize=True
                        )
                    ).add_to(m_bayes)
                    
                    # CircleMarkers sobre subcircuitos de las dos clases más altas (Clase 3 y 4)
                    for _, row in gdf_map_bayes[gdf_map_bayes['_clase_num'].isin([3, 4])].iterrows():
                        if row.geometry is None:
                            continue
                        cent = row.geometry.centroid
                        rr_val = float(row['rr_mediana_nuts'])
                        rad = 4.0 + 2.2 * rr_val
                        c_fill = COLOR_CLASE[row['_clase_num']]
                        folium.CircleMarker(
                            location=[cent.y, cent.x],
                            radius=rad,
                            color="#ffffff",
                            weight=1,
                            fill_color=c_fill,
                            fill_opacity=0.85
                        ).add_to(m_bayes)
                        
                    # Números 1 a 5 para los 5 subcircuitos con mayor rr_mediana_nuts
                    top5_df = gdf_map_bayes.sort_values('rr_mediana_nuts', ascending=False).head(5)
                    for rank_num, (_, row) in enumerate(top5_df.iterrows(), 1):
                        if row.geometry is None:
                            continue
                        cent = row.geometry.centroid
                        folium.Marker(
                            location=[cent.y, cent.x],
                            icon=folium.DivIcon(
                                html=f'<div style="font-size: 11px; font-weight: bold; color: #ffffff; text-align: center; width: 20px; height: 20px; line-height: 20px; pointer-events: none;">{rank_num}</div>',
                                icon_size=(20, 20),
                                icon_anchor=(10, 10)
                            )
                        ).add_to(m_bayes)
                        
                    # MiniMap para el cantón completo en la esquina inferior izquierda
                    MiniMap(
                        tile_layer=folium.TileLayer(
                            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                            attr="Esri"
                        ),
                        position="bottomleft",
                        zoom_level_offset=-4,
                        toggle_display=True,
                        minimized=False
                    ).add_to(m_bayes)
                    
                    # Mostrar mapa con altura de 640 px
                    st_folium(
                        m_bayes,
                        height=640,
                        use_container_width=True,
                        returned_objects=[],
                        key=f"mapa_prob_{st.session_state.get('reset', 0)}"
                    )
                    
                with col_panel:
                    total_sub = len(gdf_map_bayes)
                    c4 = clases_info[0][3]
                    c3 = clases_info[1][3]
                    c2 = clases_info[2][3]
                    c1 = clases_info[3][3]
                    
                    pct4 = (c4 / total_sub) * 100
                    pct3 = (c3 / total_sub) * 100
                    pct2 = (c2 / total_sub) * 100
                    pct1 = (c1 / total_sub) * 100
                    
                    # Tarjeta a) ZONAS DE ALTA EVIDENCIA (construida sin indentación de líneas)
                    html_card_a = (
                        f'<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 0px; padding: 16px 18px; margin-bottom: 12px;">'
                        f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 12px;">ZONAS DE ALTA EVIDENCIA</div>'
                        f'<div style="display: flex; align-items: baseline; gap: 14px;">'
                        f'<span style="font-size: 40px; font-weight: 700; color: #ff4d8d; line-height: 1;">{c4}</span>'
                        f'<div>'
                        f'<div style="font-size: 15px; font-weight: 600; color: #f0f6fc; line-height: 1.2;">de {total_sub} subcircuitos</div>'
                        f'<div style="font-size: 11px; color: #8b949e; margin-top: 3px;">{subtitulo_a}</div>'
                        f'</div>'
                        f'</div>'
                        f'<div style="display: flex; width: 100%; height: 7px; border-radius: 0px; overflow: hidden; margin-top: 14px; background-color: #21262d;">'
                        f'<div style="width: {pct4}%; background-color: #ff3d81;" title="Clase 4: {c4}"></div>'
                        f'<div style="width: {pct3}%; background-color: #ff9e3d;" title="Clase 3: {c3}"></div>'
                        f'<div style="width: {pct2}%; background-color: #7b5ea7;" title="Clase 2: {c2}"></div>'
                        f'<div style="width: {pct1}%; background-color: #232b36;" title="Clase 1: {c1}"></div>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(html_card_a, unsafe_allow_html=True)
                    
                    # Tarjeta b) CLASES DE PROBABILIDAD (construida sin indentación de líneas)
                    rows_b_html = "".join([
                        f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; font-size: 12px;">'
                        f'<div style="display: flex; align-items: center; gap: 10px; width: 105px;">'
                        f'<span style="width: 18px; height: 14px; background-color: {COLOR_CLASE[cls_id]}; border-radius: 0px; display: inline-block;"></span>'
                        f'<span style="color: #f0f6fc; font-weight: 500;">{cls_name}</span>'
                        f'</div>'
                        f'<span style="color: #8b949e; font-size: 11px; font-family: monospace; text-align: left; flex: 1; padding-left: 10px;">{cls_range}</span>'
                        f'<span style="color: #f0f6fc; font-weight: 700; font-size: 13px; text-align: right; min-width: 30px;">{cls_count}</span>'
                        f'</div>'
                        for cls_id, cls_name, cls_range, cls_count in clases_info
                    ])
                    html_card_b = (
                        f'<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 0px; padding: 16px 18px; margin-bottom: 12px;">'
                        f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 14px;">{titulo_clases}</div>'
                        f'{rows_b_html}'
                        f'</div>'
                    )
                    st.markdown(html_card_b, unsafe_allow_html=True)
                    
                    # Tarjeta c) MAYOR RIESGO RELATIVO (NUTS) (construida sin indentación de líneas)
                    top7_df = gdf_map_bayes.sort_values('rr_mediana_nuts', ascending=False).head(7)
                    rr_max_scale = 9.0
                    rows_c_html = "".join([
                        f'<div style="display: flex; align-items: center; margin-bottom: 8px; font-size: 12px;">'
                        f'<span style="color: #8b949e; font-weight: 600; width: 14px; font-size: 11px;">{str(rank_idx) if rank_idx <= 5 else "&nbsp;"}</span>'
                        f'<span style="color: #f0f6fc; font-weight: 500; font-size: 11.5px; width: 125px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{str(r["subcircuito_nom"]).title()}">{str(r["subcircuito_nom"]).title()}</span>'
                        f'<span style="color: #ff4d8d; font-weight: 700; font-size: 12px; width: 34px; text-align: right; margin-right: 12px;">{float(r["rr_mediana_nuts"]):.2f}</span>'
                        f'<div style="flex: 1; height: 14px; position: relative; min-width: 80px;">'
                        f'<div style="position: absolute; top: 6px; left: 0; width: 100%; height: 2px; background-color: #30363d; border-radius: 1px;"></div>'
                        f'<div style="position: absolute; top: 5px; left: {max(0.0, min(100.0, (float(r["rr_limite_inf_95_nuts"]) / rr_max_scale) * 100)):.1f}%; width: {max(2.0, min(100.0 - max(0.0, min(100.0, (float(r["rr_limite_inf_95_nuts"]) / rr_max_scale) * 100)), ((float(r["rr_limite_sup_95_nuts"]) - float(r["rr_limite_inf_95_nuts"])) / rr_max_scale) * 100)):.1f}%; height: 4px; background-color: #8b949e; border-radius: 2px;"></div>'
                        f'<div style="position: absolute; top: 3px; left: calc({max(0.0, min(100.0, (float(r["rr_mediana_nuts"]) / rr_max_scale) * 100)):.1f}% - 4px); width: 8px; height: 8px; background-color: #ff4d8d; border: 1.5px solid #ffffff; border-radius: 50%;"></div>'
                        f'</div>'
                        f'</div>'
                        for rank_idx, (_, r) in enumerate(top7_df.iterrows(), 1)
                    ])
                    
                    html_card_c = (
                        f'<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 0px; padding: 16px 18px;">'
                        f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 12px;">MAYOR RIESGO RELATIVO (NUTS)</div>'
                        f'{rows_c_html}'
                        f'<div style="font-size: 10.5px; color: #8b949e; margin-top: 12px; border-top: 1px solid #21262d; padding-top: 8px;">Círculo: RR mediano · barra: intervalo de credibilidad 95 %</div>'
                        f'</div>'
                    )
                    st.markdown(html_card_c, unsafe_allow_html=True)
                    
                # 3. Nota al pie (2 líneas)
                st.caption(r"NUTS: 2 cadenas, 1 000 muestras. Con ADVI las probabilidades se saturan en 1,00; por eso el mapa usa NUTS.")
                st.caption(r"Cada círculo marca un subcircuito de las dos clases más altas; su tamaño crece con el RR mediano. RR = exp(φᵢ) / media espacial de exp(φ): riesgo frente al promedio de subcircuitos, a igual año y área.")
                
        # Pestaña 6: Resumen del Modelo
        with subtab_resumen:
            st.subheader("Tabla Resumen del Modelo Posterior (az.summary)")
            st.markdown("Estadísticas resumidas e intervalos de credibilidad del 94% (HDI) para los parámetros clave del modelo generadas a partir de 2,000 muestras variacionales (ADVI).")
            
            # Generar tabla interactiva
            summary_vars = [v for v in ['alpha', 'beta_year', 'beta_area', 'sigma_s', 'rho', 'alpha_disp', 'sigma_phi', 'sigma_theta'] if v in trace.posterior]
            summary_df = az.summary(trace, var_names=summary_vars)
            st.dataframe(summary_df.style.format(precision=4), width="stretch")
            st.markdown("""
            *Nota Metodológica:*
            - **mean / sd:** Media y desviación estándar posterior estimada por aproximación variacional ADVI.
            - **hdi_3% / hdi_97%:** Intervalo de credibilidad de densidad posterior más alta al 94% (HDI).
            - **mcse_mean / mcse_sd:** Error estándar Monte Carlo de las 2,000 extracciones variacionales.
            - Para la comparación rigurosa con cadenas MCMC y el factor de reducción de escala $\\hat{R}$, consultar la **Pestaña 4: Validación ADVI vs. NUTS**.
            """)
            
            st.info("""
            ⚠️ **Nota Metodológica sobre Dinámica Espacio-Temporal y Hotspots:**
            El modelo de referencia posee una estructura puramente aditiva: $\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t=2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i$.
            Al no incluir un término de interacción espacio-temporal $\\delta_{it}$ (Knorr-Held, 2000), el riesgo relativo espacial de cada subcircuito es matemáticamente constante en el tiempo ($RR_{i, 2024} = RR_{i, 2025}$, ej. Guasmo 1 tiene 4.077 en ambos años).
            Por lo tanto, la categoría de 'hotspot persistente' no refleja trayectorias locales dinámicas y se presenta formalmente en la tesis como una limitación metodológica del modelo aditivo.
            """)

# ================= TAB 4: SIMULADOR TÁCTICO DE RIESGO =================
with tab_sim:
    # 1. Datos del simulador (función @st.cache_data propia)
    @st.cache_data
    def cargar_datos_simulador():
        csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
        if not os.path.exists(csv_path):
            csv_path = "outputs/data/resultados_riesgo_relativo.csv"
        if not os.path.exists(csv_path):
            return None
        return pd.read_csv(csv_path)

    df_rr_sim = cargar_datos_simulador()
    cols_nuts = [
        'prob_exced_1_5_nuts', 'prob_exced_1_0_nuts',
        'rr_mediana_nuts', 'rr_limite_inf_95_nuts', 'rr_limite_sup_95_nuts', 'rr_estimado_mediana'
    ]
    if df_rr_sim is None or not all(c in df_rr_sim.columns for c in cols_nuts):
        st.error("No se encontraron los resultados NUTS (resultados_riesgo_relativo.csv).")
        st.stop()

    cols_merge = ['cod_subcircu'] + cols_nuts
    base_cols = [c for c in gdf_subcircuitos.columns if c not in cols_nuts or c == 'cod_subcircu']
    gdf_sim = gdf_subcircuitos[base_cols].copy().merge(df_rr_sim[cols_merge], on='cod_subcircu', how='inner')
    if 'subcircuito_nom' not in gdf_sim.columns and 'subcircuito_nom' in df_rr_sim.columns:
        gdf_sim['subcircuito_nom'] = df_rr_sim['subcircuito_nom']

    # Preformatear columnas para tooltip
    gdf_sim['Subcircuito'] = gdf_sim['subcircuito_nom'].astype(str).str.title()
    gdf_sim['P(RR>1,5) NUTS'] = gdf_sim['prob_exced_1_5_nuts'].map(lambda x: f"{x:.3f}")
    gdf_sim['RR mediano NUTS'] = gdf_sim['rr_mediana_nuts'].map(lambda x: f"{x:.2f}")

    # 4. Textos introductorios
    st.subheader("🎯 Simulador de Riesgo Bayesiano Geográfico")
    st.markdown("Haz clic en el mapa para ver el riesgo relativo y las probabilidades posteriores del subcircuito correspondiente.")

    col_s1, col_s2 = st.columns([2, 1])

    with col_s1:
        # Bounding box urbano Guayaquil (EPSG:32717, polígonos < 2 km2, cuantiles 0.02 y 0.98 + 8% margen)
        from shapely.geometry import box

        gdf_proj = gdf_sim.to_crs("EPSG:32717")
        urban_cents = gdf_proj[gdf_proj.geometry.area < 2_000_000].geometry.centroid
        x_min, x_max = urban_cents.x.quantile(0.02), urban_cents.x.quantile(0.98)
        y_min, y_max = urban_cents.y.quantile(0.02), urban_cents.y.quantile(0.98)
        dx, dy = x_max - x_min, y_max - y_min
        bbox_poly = box(x_min - 0.08 * dx, y_min - 0.08 * dy, x_max + 0.08 * dx, y_max + 0.08 * dy)
        bbox_s = gpd.GeoSeries([bbox_poly], crs="EPSG:32717").to_crs("EPSG:4326")
        minx, miny, maxx, maxy = bbox_s.total_bounds

        # Mapa con base Esri Canvas Dark Gray (sin API key)
        m_sim = folium.Map(
            tiles=None,
            location=[-2.18, -79.90],
            zoom_start=12,
            control_scale=True
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Dark Gray"
        ).add_to(m_sim)

        def sim_style_function(feature):
            p15 = feature['properties'].get('prob_exced_1_5_nuts', 0.0)
            if p15 is None:
                p15 = 0.0
            if p15 >= 0.90:
                fill_color = '#ff3d81'
                fill_op = 0.55
            elif p15 >= 0.50:
                fill_color = '#ff9e3d'
                fill_op = 0.55
            elif p15 >= 0.10:
                fill_color = '#7b5ea7'
                fill_op = 0.55
            else:
                fill_color = '#232b36'
                fill_op = 0.90
            return {
                'fillColor': fill_color,
                'color': '#0d1117',
                'weight': 0.5,
                'fillOpacity': fill_op
            }

        folium.GeoJson(
            gdf_sim,
            style_function=sim_style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['Subcircuito', 'P(RR>1,5) NUTS', 'RR mediano NUTS'],
                aliases=['Subcircuito:', 'P(RR>1,5) NUTS:', 'RR mediano NUTS:'],
                localize=True
            )
        ).add_to(m_sim)

        m_sim.fit_bounds([[miny, minx], [maxy, maxx]])

        # Capturar clics del usuario
        click_data = st_folium(m_sim, height=580, use_container_width=True, key="sim_map", returned_objects=["last_clicked"])

    with col_s2:
        def fmt_pct(val):
            pct = val * 100.0
            if abs(pct - 100.0) < 1e-5:
                return "100 %"
            if abs(pct) < 1e-5:
                return "0 %"
            return f"{pct:.1f} %".replace(".", ",")

        punto_objetivo = click_data.get("last_clicked") if click_data else None

        if punto_objetivo:
            lat_c = punto_objetivo["lat"]
            lng_c = punto_objetivo["lng"]
            point_geom = Point(lng_c, lat_c)

            contiene = gdf_sim[gdf_sim.geometry.contains(point_geom)]
            if not contiene.empty:
                zona_cercana = contiene.iloc[0]
                es_fuera = False
            else:
                dists = gdf_sim.geometry.distance(point_geom)
                zona_cercana = gdf_sim.loc[dists.idxmin()]
                es_fuera = True

            p15 = float(zona_cercana['prob_exced_1_5_nuts'])
            p10 = float(zona_cercana['prob_exced_1_0_nuts'])
            p_bajo = max(0.0, 1.0 - p10)
            rr_val = float(zona_cercana['rr_mediana_nuts'])
            lo_val = float(zona_cercana['rr_limite_inf_95_nuts'])
            hi_val = float(zona_cercana['rr_limite_sup_95_nuts'])

            p_str = fmt_pct(p15)

            if p15 >= 0.90:
                color = "#ff3d81"
                titulo_clase = "EVIDENCIA MUY ALTA DE RIESGO ELEVADO"
                texto_explicacion = f"La probabilidad posterior de que el riesgo relativo supere 1,5 es de {p_str}. Es uno de los subcircuitos con mayor evidencia de riesgo por encima del promedio."
            elif p15 >= 0.50:
                color = "#ff9e3d"
                titulo_clase = "EVIDENCIA ALTA"
                texto_explicacion = f"La probabilidad posterior de que el riesgo relativo supere 1,5 es de {p_str}. Hay indicios de riesgo elevado, aunque con incertidumbre."
            elif p15 >= 0.10:
                color = "#7b5ea7"
                titulo_clase = "EVIDENCIA MODERADA"
                texto_explicacion = f"La probabilidad posterior de que el riesgo relativo supere 1,5 es de {p_str}. No hay evidencia suficiente para afirmar un riesgo elevado."
            else:
                color = "#8b949e"
                titulo_clase = "EVIDENCIA BAJA"
                texto_explicacion = f"La probabilidad posterior de que el riesgo relativo supere 1,5 es de {p_str}. El riesgo estimado no supera de forma clara el promedio."

            subcircuito_nombre = str(zona_cercana.get('subcircuito_nom', '')).title()
            aviso_fuera = ""
            if es_fuera:
                aviso_fuera = '<div style="font-size: 11px; color: #9aa4b2; margin-top: 3px;">Clic fuera de un subcircuito: se muestra el más cercano.</div>'

            p_bajo_str = fmt_pct(p_bajo)
            p_alto_str = fmt_pct(p10)
            p_crit_str = fmt_pct(p15)

            pct_bajo = max(0.0, min(100.0, p_bajo * 100.0))
            pct_alto = max(0.0, min(100.0, p10 * 100.0))
            pct_crit = max(0.0, min(100.0, p15 * 100.0))

            resultados_html = (
                f'<div style="background-color: #161b22; border: 1px solid #30363d; border-left: 5px solid {color}; border-radius: 0px; padding: 16px; margin-bottom: 16px;">'
                f'<div style="font-size: 11px; font-weight: 700; color: #9aa4b2; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 8px;">'
                f'EVALUACIÓN BAYESIANA DE RIESGO'
                f'</div>'
                f'<div style="font-size: 13px; color: #9aa4b2; margin-bottom: 14px;">'
                f'Subcircuito: <b style="color: #58a6ff; font-size: 14px;">{subcircuito_nombre}</b>'
                f'{aviso_fuera}'
                f'</div>'
                f'<div style="background-color: #0d1117; border: 1px solid #30363d; border-radius: 0px; padding: 12px; margin-bottom: 16px;">'
                f'<div style="font-size: 11px; color: #9aa4b2; text-transform: uppercase; letter-spacing: 0.5px;">Clasificación de Riesgo Posterior:</div>'
                f'<div style="color: {color}; font-size: 13px; font-weight: 700; margin-top: 4px; line-height: 1.3;">{titulo_clase}</div>'
                f'<div style="font-size: 12px; color: #e6edf3; margin-top: 6px; line-height: 1.45;">{texto_explicacion}</div>'
                f'</div>'
                f'<div style="display: flex; flex-direction: column; gap: 10px;">'
                f'<div>'
                f'<div style="display: flex; justify-content: space-between; font-size: 12px; color: #9aa4b2; margin-bottom: 4px;">'
                f'<span>P(RR &lt; 1,0)</span>'
                f'<b style="color: #7b5ea7;">{p_bajo_str}</b>'
                f'</div>'
                f'<div style="height: 6px; background-color: #21262d; border-radius: 0px; overflow: hidden;">'
                f'<div style="width: {pct_bajo}%; height: 100%; background-color: #7b5ea7; border-radius: 0px;"></div>'
                f'</div>'
                f'</div>'
                f'<div>'
                f'<div style="display: flex; justify-content: space-between; font-size: 12px; color: #9aa4b2; margin-bottom: 4px;">'
                f'<span>P(RR &gt; 1,0)</span>'
                f'<b style="color: #ff9e3d;">{p_alto_str}</b>'
                f'</div>'
                f'<div style="height: 6px; background-color: #21262d; border-radius: 0px; overflow: hidden;">'
                f'<div style="width: {pct_alto}%; height: 100%; background-color: #ff9e3d; border-radius: 0px;"></div>'
                f'</div>'
                f'</div>'
                f'<div>'
                f'<div style="display: flex; justify-content: space-between; font-size: 12px; color: #9aa4b2; margin-bottom: 4px;">'
                f'<span>P(RR &gt; 1,5)</span>'
                f'<b style="color: #ff3d81;">{p_crit_str}</b>'
                f'</div>'
                f'<div style="height: 6px; background-color: #21262d; border-radius: 0px; overflow: hidden;">'
                f'<div style="width: {pct_crit}%; height: 100%; background-color: #ff3d81; border-radius: 0px;"></div>'
                f'</div>'
                f'</div>'
                f'</div>'
                f'<div style="margin-top: 14px; border-top: 1px solid #30363d; padding-top: 10px; font-size: 12px; color: #e6edf3;">'
                f'RR mediano (NUTS): {rr_val:.2f} · IC 95 %: [{lo_val:.2f}; {hi_val:.2f}]'
                f'</div>'
                f'</div>'
            )
            st.markdown(resultados_html, unsafe_allow_html=True)

            # Ecuaciones matemáticas en Latex
            st.markdown("##### Formulación matemática del modelo (b): Binomial Negativo + ICAR")
            st.latex(r"\log(\mu_{it}) = \alpha + \beta_{\text{year}}\,\mathbb{I}(t=2025) + \beta_{\text{area}}\log(\text{Área}_i) + \phi_i")
            st.latex(r"\phi \sim \text{ICAR}(\sigma_\phi),\qquad Y_{it}\sim \text{NB}(\mu_{it},\alpha_{\text{disp}})")
            st.markdown(r"""
- $\phi_i$: efecto espacial ICAR con contigüidad de reina.
- $\sigma_\phi \sim \text{Exp}(1)$.
- $\beta_{\text{year}}$: efecto de 2025 frente a 2024, prior $\text{Normal}(0,1)$.
- $\beta_{\text{area}}$: coeficiente de $\log(\text{Área})$, prior $\text{Normal}(0,1)$.
- $\alpha_{\text{disp}}$: parámetro de dispersión, prior $\text{Exp}(1)$.

El modelo BYM2 se evaluó como modelo comparativo (d); ver Pestaña Comparación.
""")
        else:
            esperando_html = (
                '<div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 0px; padding: 24px; text-align: center;">'
                '<div style="font-size: 11px; font-weight: 700; color: #9aa4b2; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 12px;">ESPERANDO CLIC</div>'
                '<p style="color: #e6edf3; font-size: 13px; line-height: 1.5; margin: 0;">Por favor, haz <b>clic en cualquier punto del mapa a la izquierda</b> para obtener las probabilidades posteriores de riesgo.</p>'
                '</div>'
            )
            st.markdown(esperando_html, unsafe_allow_html=True)

# ================= TAB 5: DOWNLOADS =================
with tab_downloads:
    st.subheader("Descarga de Reportes y Datos para la Tesis")
    st.markdown("Usa los siguientes botones para descargar las bases de datos procesadas y el reporte consolidado listo para tu tesis.")
    
    col_dw1, col_dw2 = st.columns(2)
    
    with col_dw1:
        st.markdown("### 📝 Reporte Word Formateado (Norma APA)")
        st.markdown("Descarga el documento Word `.docx` con todas las tablas numeradas e integradas de acuerdo con la norma de presentación de tesis.")
        
        word_path = os.path.join(config.TABLES_DIR, "Reporte_Tesis_Flujo_Bayesiano.docx")
        if os.path.exists(word_path):
            with open(word_path, "rb") as f:
                btn = st.download_button(
                    label="📥 Descargar Reporte de Tesis en Word",
                    data=f,
                    file_name="Reporte_Tesis_Flujo_Bayesiano.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        else:
            st.warning("⚠️ El archivo Word aún no ha sido generado. Ejecuta primero `run_pipeline.py` en la consola para crearlo.")
            
    with col_dw2:
        st.markdown("### 📊 Datos Agregados y Limpios")
        st.markdown("Descarga el archivo CSV consolidado con los homicidios agregados por subcircuito y año, incluyendo la tasa y el riesgo relativo.")
        
        csv_path = os.path.join(config.CLEAN_DATA_DIR, "resultados_riesgo_relativo.csv")
        if os.path.exists(csv_path):
            df_download = pd.read_csv(csv_path)
            st.download_button(
                label="📥 Descargar Base de Riesgos en CSV",
                data=df_download.to_csv(index=False),
                file_name="resultados_riesgo_relativo.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ La base de datos de riesgos relativos aún no ha sido generada. Ejecuta primero `run_pipeline.py` en la consola.")
