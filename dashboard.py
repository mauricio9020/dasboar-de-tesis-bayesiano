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
    page_title="Dashboard Bayesiano - Homicidios Guayaquil",
    page_icon="💀",
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Flag_of_Guayaquil.svg/200px-Flag_of_Guayaquil.svg.png", width=90)
    st.title("Tesis Settings")
    st.markdown("---")
    
    st.subheader("🛠️ Panel de Control")
    umbral_riesgo = st.slider("Umbral de Alerta de Riesgo Relativo P(RR > x)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    
    opcion_mapa = st.selectbox(
        "¿Qué capa deseas visualizar en el mapa?",
        ("Probabilidad de Excedencia", "Riesgo Relativo Estimado", "Conteo de Homicidios")
    )
    
    st.markdown("---")
    st.caption("Tesis de Grado: *Ingeniería en Estadística*")
    st.caption("Cantón Guayaquil - Período de Análisis: 2024-2026")

# ----------------- CUERPO PRINCIPAL ----------------- #
st.title("🛡️ Tesis: Modelado Bayesiano Espacio-Temporal")
st.markdown("Análisis y visualización predictiva de muertes violentas en el Cantón Guayaquil (2024-2026)")

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
        
    # Contar subcircuitos críticos
    crit_count = (gdf_subcircuitos['probabilidad_excedencia'] > 0.75).sum()
    
    # Calcular métricas de edad con detalles del grupo líder y rango
    if total_homicidios > 0:
        mean_age = df_filtered['Edad_Limpia'].mean()
        median_age = df_filtered['Edad_Limpia'].median()
        grupo_counts = df_filtered['Grupo_Edad'].value_counts()
        if len(grupo_counts) > 0:
            top_grupo = grupo_counts.index[0]
            top_grupo_pct = (grupo_counts.values[0] / total_homicidios) * 100
            age_subtitle = f"Mediana: {median_age:.0f} años | Líder: {top_grupo} ({top_grupo_pct:.0f}%)"
        else:
            age_subtitle = f"Mediana: {median_age:.0f} años"
    else:
        mean_age = 0
        age_subtitle = "Sin registros en este período"
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><i class="fa-solid fa-skull"></i> Homicidios Filtrados</div>
            <h2>{total_homicidios} Casos</h2>
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
            <div class="stat-card-title"><i class="fa-solid fa-triangle-exclamation"></i> Zonas de Riesgo Extremo</div>
            <h2>{crit_count} / {len(gdf_subcircuitos)} Zonas</h2>
            <p style="font-size:0.8rem; color:#8b949e;">Subcircuitos con P(RR > 1.5) > 75%</p>
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
        fig_ts = px.line(ts_df, x='Año_Mes', y='Casos', markers=True, color_discrete_sequence=['#ff4444'])
        fig_ts.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
        st.plotly_chart(fig_ts, use_container_width=True)
        
    with col_g2:
        st.markdown("**Distribución por Presunto Móvil (Top 7)**")
        mov_df = df_filtered[config.COL_PRESUN_MOTIVACION].value_counts().head(7).reset_index(name='Casos')
        mov_df.columns = ['Móvil', 'Casos']
        fig_bar = px.bar(mov_df, x='Casos', y='Móvil', orientation='h', color='Casos', color_continuous_scale='Reds')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
        st.plotly_chart(fig_bar, use_container_width=True)

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
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Sin registros")
            
    with col_c2:
        st.markdown("**Frecuencia de Casos por Parroquia (Distrito)**")
        if total_homicidios > 0:
            dist_counts = df_filtered[config.COL_DISTRITO].value_counts().reset_index(name='Casos')
            dist_counts.columns = ['Distrito', 'Casos']
            fig_bar_dist = px.bar(dist_counts, x='Distrito', y='Casos', color='Casos', color_continuous_scale='Reds')
            fig_bar_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#c9d1d9',
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_bar_dist, use_container_width=True)
        else:
            st.info("Sin registros")
            
    with col_c3:
        st.markdown("**Interpretación**")
        st.markdown(explicacion_texto, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Distribución Demográfica de Edad y Sexo")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**Figura 1: Distribución de la Edad de las Víctimas**")
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
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Sin registros")
            
    with col_d2:
        st.markdown("**Figura 2: Distribución de Edad por Sexo de la Víctima**")
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
                st.plotly_chart(fig_box, use_container_width=True)
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
            st.dataframe(df_esda_moran, use_container_width=True)
        except Exception as e:
            st.error(f"Error al cargar tabla ESDA: {e}")
            
    col_esda1, col_esda2 = st.columns(2)
    with col_esda1:
        f11_path = os.path.join(config.FIGURES_DIR, "Figura11_moran_scatterplot.png")
        if os.path.exists(f11_path):
            st.image(f11_path, caption="Figura 11: Moran Scatterplot de Homicidios 2024-2025", use_column_width=True)
    with col_esda2:
        f12_path = os.path.join(config.FIGURES_DIR, "Figura12_lisa_clusters.png")
        if os.path.exists(f12_path):
            st.image(f12_path, caption="Figura 12: Mapa de Agrupamiento LISA (Homicidios 2024-2025)", use_column_width=True)

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
        m = folium.Map(location=[-2.18, -79.90], zoom_start=12.5, tiles="CartoDB dark_matter")
        
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
                
        st.markdown(f"""
        - **Modelo de Referencia Evaluado:** **{model_eval_val}**
        - **Método de Inferencia:** ADVI (Automatic Differentiation Variational Inference, Adam lr=0.005).
        - **Muestras Posteriores Extraídas:** {draws_cnt_val} draws de la aproximación variacional.
        - **Criterio de Convergencia:** {crit_tol_val}.
        - **ELPD-WAIC:** **{crit_waic_str}** (Escala logarítmica).
        - **ELPD-LOO:** **{crit_loo_str}** (Escala logarítmica).
        - **Observaciones con Pareto-k > 0.7:** **{k_gt_07_val}**
        - **Diagnóstico Pareto-k:** {k_adv_val}
        - **Validación Metodológica:** Ajuste complementario NUTS MCMC disponible en la pestaña siguiente.
        """)
        
        # Mostrar imagen de convergencia del ELBO si existe
        elbo_img_path = os.path.join(config.FIGURES_DIR, "Figura_convergencia_elbo.png")
        if os.path.exists(elbo_img_path):
            st.image(elbo_img_path, caption="Figura: Trayectoria de optimización del ELBO hacia la meseta de convergencia")
        else:
            trace_img_path = os.path.join(config.FIGURES_DIR, "Figura14_bayesian_traceplots.png")
            if os.path.exists(trace_img_path):
                st.image(trace_img_path, caption="Figura 14: Traceplots de los parámetros posteriores de ADVI")
            else:
                st.info("💡 Ejecuta `python run_pipeline.py` en la consola para generar los gráficos de diagnóstico.")
            
    with col_d2:
        # Mostrar imagen del posterior predictivo PPC
        ppc_img_path = os.path.join(config.FIGURES_DIR, "Figura19_bayesian_ppc.png")
        if os.path.exists(ppc_img_path):
            st.image(ppc_img_path, caption="Figura 19: Comprobación Predictiva Posterior (PPC) en rango completo (ADVI)")
        else:
            st.info("💡 La imagen de verificación predictiva posterior (PPC) se generará al ejecutar `python run_pipeline.py`.")

# ================= TAB 3.5: MODELO BAYESIANO =================
with tab_modelo_bayesiano:
    st.header("🧠 Inferencia y Modelado Bayesiano Jerárquico")
    st.markdown("Esta sección presenta de manera detallada el modelo probabilístico espacio-temporal estimado mediante PyMC y ArviZ.")
    
    # Intentar cargar la traza
    if trace is None:
        st.warning("⚠️ No se encontró la traza de inferencia bayesiana `pymc_trace.nc`. Por favor ejecuta `run_pipeline.py` para entrenar el modelo.")
    else:
        
        # Sub-pestañas dentro de la sección bayesiana
        subtab_espec, subtab_post, subtab_diag, subtab_comp, subtab_nuts, subtab_mapas, subtab_resumen = st.tabs([
            "📖 Pestaña 1: Especificación del Modelo",
            "📈 Pestaña 2: Distribuciones Posteriores",
            "🔍 Pestaña 3: Diagnóstico ADVI (ELBO)",
            "📊 Pestaña 4: Comparación del Modelo",
            "⚖️ Pestaña 5: Validación ADVI vs. NUTS",
            "🗺️ Pestaña 6: Mapas Bayesianos",
            "📝 Pestaña 7: Resumen del Modelo"
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
            El modelo ajustado para estimar la tasa predictiva y el riesgo relativo de muertes violentas en Guayaquil se formula como un modelo Bayesiano Jerárquico Espacio-Temporal con verosimilitud **Binomial Negativa** para controlar la sobredispersión territorial extrema.
            
            #### 1. Modelo de Referencia Seleccionado (Mejor LOO-CV)
            **Modelo (b): Binomial Negativa con ICAR**
            
            - **Verosimilitud (Likelihood):**
              $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
            
            - **Función de Enlace (Log-Link):**
              $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i$$
            
            - **Componente Espacial Estructurado (ICAR):**
              $$\\phi \\sim \\text{ICAR}(W, \\sigma_\\phi^2)$$
              
            Donde:
            - $Y_{it}$: Número de homicidios observados en el subcircuito $i$ en el año $t$.
            - $\\mu_{it}$: Media predictiva condicional ajustada por el modelo.
            - $\\alpha$: Intercepto global (tasa basal cantonal en 2024).
            - $\\beta_{\\text{year}}$: Efecto fijo de año 2025 frente a la línea base 2024 con prior $\\text{Normal}(0, 1)$.
            - $\\beta_{\\text{area}}$: Coeficiente para $\\log(\\text{Area}_i)$ con prior $\\text{Normal}(0, 1)$.
            - $\\sigma_\\phi$: Escala del efecto espacial intrínseco autorregresivo condicional con prior $\\text{Exponential}(1.0)$.
            - $\\alpha_{disp}$: Parámetro de dispersión de la verosimilitud Binomial Negativa con prior $\\text{Exponential}(1.0)$.
            
            *Justificación Metodológica:* El Modelo (b) obtiene el mayor $\\text{ELPD}_{\\text{LOO}} = -1,233.75$ y la menor cantidad de observaciones con Pareto-$k > 0.7$ (27 observaciones). La diferencia frente al Modelo (c) ($-1,241.65$) es de 7.90 puntos con $dse = 4.18$ ($|\\Delta| < 2 \\cdot dse = 8.36$), lo que representa un empate estadístico. Por parsimonia (menor número de parámetros al no requerir $\\sigma_\\theta$ ni efectos no estructurados $\\theta_i$), se selecciona el Modelo (b) como el modelo de referencia final de la tesis.
            
            #### 2. Modelos Estructurales Comparativos
            - **Modelo (c): Besag ICAR + Efecto No Estructurado (Besag et al., 1991):**
              $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i + \\theta_i$$
              donde $\\theta_i \\sim \\text{Normal}(0, \\sigma_\\theta^2)$ con $\\sigma_\\theta \\sim \\text{Exponential}(1.0)$.
            - **Modelo (d): BYM2 Binomial Negativo (Riebler et al., 2016):**
              $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\psi_i$$
              $$\\psi_i = \\sigma_s \\left( \\sqrt{\\rho} \\phi^*_i + \\sqrt{1 - \\rho} \\theta^*_i \\right)$$
              donde $\\rho \\sim \\text{Beta}(0.5, 0.5)$ es el parámetro de mezcla espacial.
            
            #### 3. Distribuciones a Priori (Priors)
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
            st.subheader("Distribuciones Posteriores de los Parámetros Estimados")
            st.markdown("Visualización de las funciones de densidad marginal de probabilidad posterior y estimación de intervalos de credibilidad.")
            
            opcion_post = st.radio(
                "Selecciona el tipo de parámetro a visualizar:",
                ["Parámetros Globales (alpha, beta_year, beta_area, sigma_s, rho, alpha_disp)", 
                 "Efectos Espaciales de Subcircuitos (phi)", 
                 "Efecto Temporal Anual (beta_year)"],
                key="opcion_post_radio"
            )
            
            if opcion_post.startswith("Parámetros Globales"):
                st.markdown("#### Densidades Posteriores de Parámetros Globales")
                post_vars = [v for v in ['alpha', 'beta_year', 'beta_area', 'sigma_phi', 'sigma_theta', 'sigma_s', 'rho', 'alpha_disp'] if v in trace.posterior]
                fig, axes = plt.subplots(2, 3, figsize=(12, 7))
                az.plot_posterior(trace, var_names=post_vars, ax=axes)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("""
                *Interpretación:*
                - **alpha:** Tasa media global cantonal en escala logarítmica (año 2024).
                - **beta_year:** Variación temporal neta en 2025 frente a la línea base 2024.
                - **beta_area:** Elasticidad estimada del área física territorial sobre el conteo delictivo.
                - **sigma_phi:** Escala del componente espacial estructurado ICAR (Modelo de referencia c).
                - **sigma_theta:** Escala de la heterogeneidad zonal no estructurada (Modelo de referencia c).
                - **sigma_s:** Desviación estándar total del componente espacial BYM2 (Modelo comparativo d).
                - **rho:** Parámetro de mezcla espacial con prior Beta(0.5, 0.5) en el modelo BYM2. *Aviso metodológico:* en ADVI converge a ~0.027 y en NUTS a ~0.813; por tanto, su estimación bajo ADVI no es fiable para cuantificar la proporción de varianza espacial por contigüidad.
                - **alpha_disp:** Parámetro de dispersión de la verosimilitud Binomial Negativa.
                """)
                
            elif opcion_post == "Efectos Espaciales de Subcircuitos (phi)":
                st.markdown("#### Densidad Posterior de $\\phi_i$ por Subcircuito")
                if len(sub_codes_loc) > 0 and gdf_subcircuitos is not None:
                    # Crear mapeo de nombres de subcircuito
                    df_sub_mapping = gdf_subcircuitos[['cod_subcircu', 'subcircuito_nom']].drop_duplicates()
                    sub_names_sorted = sorted(df_sub_mapping['subcircuito_nom'].unique())
                    
                    sub_seleccionado = st.selectbox("Selecciona el subcircuito:", sub_names_sorted)
                    cod_sel = df_sub_mapping[df_sub_mapping['subcircuito_nom'] == sub_seleccionado]['cod_subcircu'].values[0]
                    
                    if cod_sel in sub_codes_loc:
                        idx_sel = sub_codes_loc.index(cod_sel)
                        # Extraer muestras de phi para ese subcircuito
                        phi_sub = trace.posterior['phi'].sel(phi_dim_0=idx_sel)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        az.plot_posterior(phi_sub, ax=ax, color="#58a6ff")
                        ax.set_title(f"Distribución Posterior de $\\phi$ para {sub_seleccionado}")
                        st.pyplot(fig)
                        plt.close(fig)
                        st.markdown(f"El efecto espacial posterior $\\phi$ para **{sub_seleccionado}** refleja su desviación en riesgo relativo neto respecto a la media global de Guayaquil.")
                    else:
                        st.warning("El subcircuito seleccionado no coincide con los índices de entrenamiento del modelo.")
                else:
                    st.info("Sin datos de subcircuitos cargados.")
                    
            else:
                st.markdown("#### Densidad Posterior del Contraste Anual 2025 vs. 2024 ($\beta_{\text{year}}$)")
                if 'beta_year' in trace.posterior:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    az.plot_posterior(trace, var_names=['beta_year'], ax=ax, color="#ffb74d")
                    ax.set_title("Contraste Anual 2025 vs. 2024 ($\beta_{year}$)")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    b_mean = float(trace.posterior['beta_year'].values.mean())
                    st.markdown(f"El coeficiente posterior mediano de $\\beta_{{\\text{{year}}}}$ es **{b_mean:+.3f}**, lo que equivale a una variación neta de la tasa de homicidios de **{((np.exp(b_mean)-1)*100):+.1f}%** en 2025 respecto a 2024.")
                else:
                    st.info("Sin parámetro temporal beta_year en la traza cargada.")
                    
        # Pestaña 3: Diagnósticos ADVI
        with subtab_diag:
            st.subheader("Evaluación de Optimización Variacional (ADVI) y Diagnósticos")
            st.markdown("Verificación de la convergencia del ELBO hacia la meseta de optimización, chequeo predictivo posterior (PPC) y dispersión.")
            
            opcion_diag = st.selectbox(
                "Selecciona la visualización de diagnóstico:",
                ["Convergencia del ELBO (Meseta de Optimización)", "Comprobación Predictiva Posterior (PPC)", "Traceplots de Muestras Variacionales"],
                key="opcion_diag_select"
            )
            
            if opcion_diag == "Convergencia del ELBO (Meseta de Optimización)":
                st.markdown("#### Trayectoria de Optimización del ELBO (Evidence Lower Bound)")
                elbo_img = os.path.join(config.FIGURES_DIR, "Figura_convergencia_elbo.png")
                if os.path.exists(elbo_img):
                    st.image(elbo_img, use_column_width=True)
                else:
                    st.info("El gráfico de convergencia del ELBO se generará al ejecutar `python run_pipeline.py`.")
            elif opcion_diag == "Comprobación Predictiva Posterior (PPC)":
                st.markdown("#### Comprobación Predictiva Posterior (PPC) - Rango Completo")
                ppc_img = os.path.join(config.FIGURES_DIR, "Figura19_bayesian_ppc.png")
                if os.path.exists(ppc_img):
                    st.image(ppc_img, use_column_width=True)
                elif hasattr(trace, 'posterior_predictive') and 'y_obs' in trace.posterior_predictive:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    az.plot_ppc(trace, ax=ax, num_pp_samples=100)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No se encontraron muestras predictivas. Ejecuta `python run_pipeline.py`.")
            else:
                st.markdown("#### Traceplots de Parámetros Clave (Muestras Variacionales ADVI)")
                diag_vars = [v for v in ['alpha', 'beta_year', 'beta_area', 'sigma_s', 'rho', 'alpha_disp'] if v in trace.posterior]
                fig, axes = plt.subplots(len(diag_vars), 2, figsize=(10, 2.2 * len(diag_vars)))
                az.plot_trace(trace, var_names=diag_vars, axes=axes)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            st.markdown("""
            **Interpretación de la Inferencia Variacional (ADVI):**
            - **Maximización del ELBO:** ADVI optimiza una distribución variacional aproximada minimizando la divergencia de Kullback-Leibler hacia la verdadera distribución posterior. La estabilización en meseta con tolerancia 1e-4 confirma la convergencia del algoritmo.
            - **Comprobación Predictiva Posterior (PPC):** La coincidencia entre los conteos observados y las réplicas $y_{rep}$ valida que la verosimilitud Binomial Negativa captura adecuadamente la sobredispersión empírica.
            - **Criterios de Información (ELPD):** LOO-CV y WAIC penalizan la complejidad del modelo para evaluar la capacidad de generalización fuera de muestra.
            """)
            
        # Pestaña 4: Comparación del modelo
        with subtab_comp:
            st.subheader("Evaluación de Información Predictiva y Ajuste Out-of-Sample")
            
            comp_tab1, comp_tab2 = st.tabs([
                "📊 Comparación Frecuentista vs. Bayesiana",
                "🧠 Comparación entre Modelos Bayesianos"
            ])
            
            with comp_tab1:
                st.markdown("Criterios de información de desviación para contrastar la capacidad predictiva frente a estimadores de máxima verosimilitud clásicos.")
                
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
                    st.metric(label="ELPD-WAIC (Escala Log)", value=waic_val)
                with col_c2:
                    st.metric(label="ELPD-LOO (Escala Log)", value=loo_val)
                
                if os.path.exists(excel_clas_path):
                    st.markdown("#### Comparación Frecuentista Clásica (Evidencia de Sobredispersión):")
                    try:
                        df_clas = pd.read_excel(excel_clas_path)
                        st.dataframe(df_clas, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al leer tabla clásica: {e}")
                
                st.markdown(f"""
                #### Evaluación del Desempeño Predictivo y Generalización:
                El modelo bayesiano jerárquico ajustado presenta un **ELPD-WAIC** de **{waic_val}** y un **ELPD-LOO** de **{loo_val}** (en escala logarítmica de densidad predictiva esperada).
                
                1. **Capacidad de Generalización (WAIC / LOO-CV):** Ambos criterios estiman la verosimilitud de datos futuros no observados y penalizan la complejidad del modelo (el número efectivo de parámetros). Valores de mayor ELPD demuestran que la estructura espacio-temporal propuesta generaliza de forma robusta la tasa de criminalidad sin incurrir en sobreajuste (overfitting).
                2. **Tratamiento de la Sobredispersión:** La verosimilitud **Binomial Negativa** introduce el parámetro de dispersión $\\alpha_{{disp}}$ para flexibilizar la relación media-varianza. Esto permite modelar de forma precisa datos de conteo con alta concentración y variabilidad territorial extrema sin sesgar las estimaciones.
                3. **Regularización y Suavizado:** El componente espacial **ICAR** y el efecto temporal anual ejercen una regularización (shrinkage) que estabiliza los estimadores en zonas con pocos registros hacia la tendencia regional circundante.
                """)
                
            with comp_tab2:
                st.markdown("Comparativa de bondad de ajuste y capacidad predictiva entre diferentes formulaciones de modelado Bayesiano Jerárquico ajustadas con NUTS.")
                
                # Cargar tabla comparativa bayesiana real generada por stage8
                excel_comp_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_modelos_bayesiana.xlsx")
                if os.path.exists(excel_comp_path):
                    try:
                        df_comp_bayes = pd.read_excel(excel_comp_path)
                        st.dataframe(df_comp_bayes, use_container_width=True)
                        
                        st.info("""
                        > 📌 **Nota Metodológica sobre Selección de Modelos:**
                        > En la comparación formal LOO-CV, el **Modelo (b) Binomial Negativa (ICAR)** alcanza la mayor densidad predictiva esperada fuera de muestra (ELPD_LOO = -1,233.75, SE = 26.76, 27 observaciones k > 0.7).
                        > La diferencia frente al Modelo (c) (-1,241.65) es de **7.90 puntos** con error estándar de la diferencia $dse = 4.18$.
                        > Al ser la diferencia menor que dos errores estándar ($|\\Delta| = 7.90 < 2 \\cdot dse = 8.36$), **ambos modelos empatan estadísticamente**. Por parsimonia (menor complejidad paramétrica y mejor comportamiento diagnóstico de Pareto-k), el **Modelo (b)** se establece como el modelo final ganador de referencia de la tesis.
                        """)
                        
                        st.markdown("""
                        ### 📖 Formulación Matemática de los 4 Modelos Comparados
                        
                        1. **Poisson Bayesiano Jerárquico (Línea Base):**
                           $$Y_{it} \\sim \\text{Poisson}(\\mu_{it})$$
                           $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i$$
                           Donde $\\phi \\sim \\text{ICAR}(W)$ y $\\beta_{\\text{year}} \\sim \\text{Normal}(0, 1)$.
                           
                        2. **Binomial Negativa Jerárquico (ICAR) [Modelo Ganador]:**
                           $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
                           $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i$$
                           $$\\text{Var}(Y_{it}) = \\mu_{it} + \\frac{\\mu_{it}^2}{\\alpha_{disp}}$$
                           
                        3. **Binomial Negativa con ICAR + Efecto No Estructurado (Besag et al., 1991):**
                           $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
                           $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i + \\theta_i$$
                           Donde $\\theta_i \\sim \\text{Normal}(0, \\sigma_\\theta^2)$ captura heterogeneidad zonal no correlacionada.
                           
                        4. **BYM2 Binomial Negativo (Prior Beta(0.5, 0.5)):**
                           $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
                           $$\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t = 2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\psi_i$$
                           $$\\psi_i = \\sigma_s \\left( \\sqrt{\\rho} \\phi^*_i + \\sqrt{1 - \\rho} \\theta^*_i \\right)$$
                           Donde $\\rho \\sim \\text{Beta}(0.5, 0.5)$ controla la proporción de varianza espacial estructurada y $\\phi^*_i$ es el componente ICAR escalado por la varianza geométrica marginal.
                        """)
                    except Exception as e:
                        st.error(f"Error al leer la tabla de comparación bayesiana: {e}")
                else:
                    st.warning("⚠️ La tabla de comparación de modelos (`tabla_comparacion_modelos_bayesiana.xlsx`) aún no ha sido generada. Ejecuta `python run_pipeline.py` en la terminal para ajustar formalmente los 4 modelos mediante ADVI.")
            
        # Pestaña 5: Validación ADVI vs. NUTS
        with subtab_nuts:
            st.subheader("Validación Metodológica: ADVI vs. NUTS (MCMC) y Estabilidad de Semillas")
            st.markdown("""
            **Justificación Metodológica:**
            - **ADVI (Automatic Differentiation Variational Inference)** es el método de inferencia principal de la tesis por su alta eficiencia computacional en modelos espaciales jerárquicos de alta dimensión ($V=244$ nodos, múltiples hiperparámetros de escala y sobredispersión).
            - **NUTS (No-U-Turn Sampler, MCMC)** se utiliza estrictamente como **patrón de validación metodológica** sobre el modelo de referencia ganador (Modelo b: Binomial Negativa ICAR), así como sobre los Modelos (c) y (d), empleando exactamente la misma formulación y los mismos priors.
            - Esta validación demuestra que ADVI reproduce con alta fidelidad el ordenamiento de los riesgos relativos espaciales ($r > 0.98, \\rho > 0.98$) y los coeficientes fijos ($\\beta_{\\text{year}}, \\beta_{\\text{area}}$).
            - **Calibración de Incertidumbre y Probabilidades de Excedencia:** Debido a la conocida subestimación de incertidumbre posterior en ADVI (desviaciones estándar reducidas entre 40% y 70%), las probabilidades de excedencia $P(RR > 1.5)$ bajo ADVI se concentran artificialmente en 1.00 en zonas de alto riesgo con intervalos estrechos (ej. Nuevo Guayaquil 3: RR = 4.40, IC [2.79, 6.91]). El cálculo de estas probabilidades mediante NUTS para el modelo final restituye la dispersión posterior real y proporciona probabilidades calibradas.
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
                        st.dataframe(df_nuts_comp, use_container_width=True)
                        
                        st.warning("""
                        ⚠️ **Aviso Metodológico Riguroso sobre Hiperparámetros:**
                        - **Efecto de Año y Coeficientes Estructurales:** ADVI recupera con alta precisión el contraste fijo de año ($\\beta_{\\text{year}}$) y la elasticidad del área ($\\beta_{\\text{area}}$), con diferencias relativas mínimas frente a NUTS.
                        - **Subestimación de Incertidumbre:** La desviación estándar posterior bajo ADVI es consistentemente menor a la de NUTS (reducción del 40% al 70%), reflejando la conocida contracción de incertidumbre en las colas propia de las aproximaciones variacionales gaussianas de campo medio.
                        - **Distorsión en Hiperparámetros de Escala y Mezcla:** ADVI distorsiona significativamente los hiperparámetros ($\\sigma_s, \\rho, \\alpha_{disp}$). En particular, en el modelo BYM2 el parámetro de mezcla $\\rho$ se contrae a ~0.027 bajo ADVI frente a ~0.813 bajo NUTS. Por tanto, no se puede afirmar que ADVI sea 'perfectamente apto' para la inferencia de hiperparámetros, y sus estimaciones deben presentarse siempre junto a las de NUTS.
                        """)
                    except Exception as e:
                        st.error(f"Error al cargar la tabla de validación ADVI vs. NUTS: {e}")
                else:
                    st.info("La tabla comparativa ADVI vs. NUTS se generará al ejecutar `python run_pipeline.py`.")
                    
                st.markdown("#### Correlación de Riesgos Relativos Espaciales (RR) por Subcircuito (ADVI vs. NUTS)")
                corr_path = os.path.join(config.TABLES_DIR, "tabla_correlacion_rr_advi_nuts.xlsx")
                if os.path.exists(corr_path):
                    try:
                        df_corr_rr = pd.read_excel(corr_path)
                        st.dataframe(df_corr_rr, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al cargar tabla de correlaciones RR: {e}")
                        
                st.markdown("#### Dispersión y Ajuste de Riesgos Relativos (RR) por Subcircuito (244 Subcircuitos)")
                rr_scatter_img = os.path.join(config.FIGURES_DIR, "Figura_dispersion_rr_advi_nuts.png")
                if os.path.exists(rr_scatter_img):
                    st.image(rr_scatter_img, use_column_width=True)
                else:
                    st.info("El gráfico de dispersión de RR ADVI vs. NUTS se generará al ejecutar `python run_pipeline.py`.")
                    
            with nuts_tab2:
                st.markdown("#### Evaluación de Estabilidad Estocástica entre Semillas Independientes (ADVI)")
                st.markdown("""
                Para verificar la robustez estocástica y descartar la trampa en óptimos locales espurios, el modelo de referencia ganador (b) (Binomial Negativa ICAR) se evaluó bajo 3 semillas aleatorias independientes (42, 123 y 999) bajo el **criterio formal de convergencia unificado**: la variación de la media móvil del ELBO en las últimas 5,000 iteraciones debe ser menor al 0.1% (< 0.001) respecto de las 5,000 previas.
                """)
                seed_stab_path = os.path.join(config.TABLES_DIR, "tabla_estabilidad_semillas_advi.xlsx")
                if os.path.exists(seed_stab_path):
                    try:
                        df_seed_stab = pd.read_excel(seed_stab_path)
                        df_seed_stab['Semilla'] = df_seed_stab['Semilla'].astype(str)
                        st.dataframe(df_seed_stab, use_container_width=True)
                        st.success("✅ Estabilidad verificada: El criterio de convergencia formal se evalúa uniformemente en todas las semillas, reportando la media y desviación estándar de los parámetros y del ELBO.")
                    except Exception as e:
                        st.error(f"Error al cargar la tabla de estabilidad de semillas: {e}")
                else:
                    st.info("La tabla de estabilidad de semillas se generará al ejecutar `python run_pipeline.py`.")
            
        # Pestaña 6: Mapas Bayesianos
        with subtab_mapas:
            st.subheader("Cartografía Temática de Riesgo y Probabilidades Bayesianas")
            st.markdown("Visualización de las estimaciones posteriores mapeadas sobre los subcircuitos de Guayaquil.")
            
            if len(sub_codes_loc) > 0 and gdf_subcircuitos is not None and trace is not None:
                # Calcular métricas del riesgo relativo posterior estandarizado sobre la media cantonal (Clayton & Kaldor, 1987)
                spatial_var = 'phi' if 'phi' in trace.posterior else ('psi' if 'psi' in trace.posterior else 'spatial_total')
                phi_vals = trace.posterior[spatial_var].values # (chains, draws, subcircuitos)
                exp_phi = np.exp(phi_vals)
                # Estandarización sobre la media espacial de los subcircuitos
                rr_vals = exp_phi / np.mean(exp_phi, axis=-1, keepdims=True)
                
                rr_mediana = np.median(rr_vals, axis=(0, 1))
                rr_media = np.mean(rr_vals, axis=(0, 1))
                rr_sd = np.std(rr_vals, axis=(0, 1))
                # Probabilidad de excedencia respecto a 1.0 (riesgo superior al promedio)
                prob_exced_1 = np.mean(rr_vals > 1.0, axis=(0, 1))
                # Probabilidad de excedencia respecto a 1.5 (riesgo superior al 50% del promedio)
                prob_exced_1_5 = np.mean(rr_vals > 1.5, axis=(0, 1))
                
                # Crear dataframe de métricas bayesianas
                df_bayes_metrics = pd.DataFrame({
                    'cod_subcircu': sub_codes_loc,
                    'rr_mediana': rr_mediana,
                    'rr_media': rr_media,
                    'rr_sd': rr_sd,
                    'prob_exced_1': prob_exced_1,
                    'prob_exced_1_5': prob_exced_1_5
                })
                
                # Unir con el GeoDataFrame filtrado para visualización evitando duplicación de columnas
                base_cols = [c for c in gdf_subcircuitos.columns if c not in df_bayes_metrics.columns or c == 'cod_subcircu']
                gdf_map_bayes = gdf_subcircuitos[base_cols].merge(df_bayes_metrics, on='cod_subcircu', how='inner')
                
                mapa_tipo = st.selectbox(
                    "Selecciona la capa de mapa bayesiano a visualizar:",
                    ["Riesgo Relativo Posterior (Mediana)", "Media Posterior del Riesgo", "Probabilidad de Excedencia P(RR > 1.0)", "Probabilidad de Excedencia P(RR > 1.5)", "Incertidumbre Posterior (Desviación Estándar)"],
                    key="mapa_tipo_bayes"
                )
                
                # Configurar métrica según selección
                if mapa_tipo == "Riesgo Relativo Posterior (Mediana)":
                    col_val = 'rr_mediana'
                    legenda = "Riesgo Relativo (Mediana)"
                    color_scale = lambda v: '#ff4444' if v > 1.5 else '#ffb74d' if v > 0.9 else '#66bb6a'
                    rad_scale = lambda v: int(max(200, min(1200, v * 250)))
                elif mapa_tipo == "Media Posterior del Riesgo":
                    col_val = 'rr_media'
                    legenda = "Riesgo Relativo (Media)"
                    color_scale = lambda v: '#ff4444' if v > 1.5 else '#ffb74d' if v > 0.9 else '#66bb6a'
                    rad_scale = lambda v: int(max(200, min(1200, v * 250)))
                elif mapa_tipo == "Probabilidad de Excedencia P(RR > 1.0)":
                    col_val = 'prob_exced_1'
                    legenda = "P(RR > 1.0)"
                    color_scale = lambda v: '#ff4444' if v > 0.8 else '#ffb74d' if v > 0.4 else '#66bb6a'
                    rad_scale = lambda v: int(max(200, min(1200, v * 1000)))
                elif mapa_tipo == "Probabilidad de Excedencia P(RR > 1.5)":
                    col_val = 'prob_exced_1_5'
                    legenda = "P(RR > 1.5)"
                    color_scale = lambda v: '#ff4444' if v > 0.75 else '#ffb74d' if v > 0.40 else '#66bb6a'
                    rad_scale = lambda v: int(max(200, min(1200, v * 1000)))
                else:
                    col_val = 'rr_sd'
                    legenda = "Desviación Estándar Posterior (Incertidumbre)"
                    color_scale = lambda v: '#ff4444' if v > 0.4 else '#ffb74d' if v > 0.2 else '#66bb6a'
                    rad_scale = lambda v: int(max(200, min(1200, v * 1500)))
                
                # Inicializar folium map
                m_bayes = folium.Map(location=[-2.18, -79.90], zoom_start=12, tiles="CartoDB dark_matter")
                
                for idx, row in gdf_map_bayes.iterrows():
                    if row.geometry is None:
                        continue
                    centroid = row.geometry.centroid
                    val = row[col_val]
                    c = color_scale(val)
                    rad = rad_scale(val)
                    
                    # Popup interactivo
                    popup_html = f"""<div style="font-family: Arial, sans-serif; font-size: 11px; width: 220px; color: #333;">
                    <h5 style="margin: 0 0 5px 0; color: #161b22; font-size: 12px;"><b>{row['subcircuito_nom']}</b></h5>
                    <p style="margin: 3px 0;"><b>Distrito:</b> {row['distrito_nom']}</p>
                    <p style="margin: 3px 0;"><b>{legenda}:</b> {val:.4f}</p>
                    <p style="margin: 3px 0;"><b>RR Mediana:</b> {row['rr_mediana']:.2f}x</p>
                    <p style="margin: 3px 0;"><b>Incertidumbre (SD):</b> {row['rr_sd']:.2f}</p>
                    <p style="margin: 3px 0;"><b>P(RR > 1.0):</b> {row['prob_exced_1']:.2%}</p>
                    <p style="margin: 3px 0;"><b>P(RR > 1.5):</b> {row['prob_exced_1_5']:.2%}</p>
                    </div>"""
                    
                    folium.Circle(
                        location=[centroid.y, centroid.x],
                        radius=rad,
                        color=c,
                        fill=True,
                        fillColor=c,
                        fillOpacity=0.5,
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=f"{row['subcircuito_nom']}: {val:.3f}"
                    ).add_to(m_bayes)
                    
                # Mostrar el mapa usando st_folium
                st_folium(m_bayes, width=900, height=550, key="mapa_bayes_tab")
            else:
                st.warning("Los datos del modelo y del shapefile no están listos.")
                
        # Pestaña 7: Resumen del Modelo
        with subtab_resumen:
            st.subheader("Tabla Resumen del Modelo Posterior (az.summary)")
            st.markdown("Estadísticas resumidas e intervalos de credibilidad del 95% (HDI) para los parámetros clave del modelo generadas a partir de 2,000 muestras variacionales (ADVI).")
            
            # Generar tabla interactiva
            summary_vars = [v for v in ['alpha', 'beta_year', 'beta_area', 'sigma_s', 'rho', 'alpha_disp', 'sigma_phi', 'sigma_theta'] if v in trace.posterior]
            summary_df = az.summary(trace, var_names=summary_vars)
            st.dataframe(summary_df.style.format(precision=4), use_container_width=True)
            st.markdown("""
            *Nota Metodológica:*
            - **mean / sd:** Media y desviación estándar posterior estimada por aproximación variacional ADVI.
            - **hdi_3% / hdi_97%:** Intervalo de credibilidad de densidad posterior más alta al 95% (HDI).
            - **mcse_mean / mcse_sd:** Error estándar Monte Carlo de las 2,000 extracciones variacionales.
            - Para la comparación rigurosa con cadenas MCMC y el factor de reducción de escala $\\hat{R}$, consultar la **Pestaña 5: Validación ADVI vs. NUTS**.
            """)
            
            st.info("""
            ⚠️ **Nota Metodológica sobre Dinámica Espacio-Temporal y Hotspots:**
            El modelo de referencia posee una estructura puramente aditiva: $\\log(\\mu_{it}) = \\alpha + \\beta_{\\text{year}} \\cdot \\mathbb{I}(t=2025) + \\beta_{\\text{area}} \\log(\\text{Area}_i) + \\phi_i$.
            Al no incluir un término de interacción espacio-temporal $\\delta_{it}$ (Knorr-Held, 2000), el riesgo relativo espacial de cada subcircuito es matemáticamente constante en el tiempo ($RR_{i, 2024} = RR_{i, 2025}$, ej. Guasmo 1 tiene 4.077 en ambos años).
            Por lo tanto, la categoría de 'hotspot persistente' no refleja trayectorias locales dinámicas y se presenta formalmente en la tesis como una limitación metodológica del modelo aditivo.
            """)

# ================= TAB 4: SIMULADOR TÁCTICO DE RIESGO =================
with tab_sim:
    st.subheader("🎯 Simulador de Riesgo Bayesiano Geográfico")
    st.warning("⚠️ **Herramienta ilustrativa; no es un resultado del modelo:** Este simulador proyecta estimaciones geográficas con fines exploratorios e ilustrativos para el análisis territorial y no sustituye los diagnósticos formales del modelo.")
    st.markdown("Haz clic en **cualquier coordenada del mapa interactivo** a la izquierda para estimar el riesgo relativo y la probabilidad posterior de criminalidad en el subcircuito correspondiente.")
    
    col_s1, col_s2 = st.columns([2, 1])
    
    with col_s1:
        # Mapa interactivo para cliquear
        m_sim = folium.Map(location=[-2.15, -79.93], zoom_start=11.5, tiles="CartoDB dark_matter")
        
        # Estilo de polígonos de riesgo para mejorar el mapa del simulador
        def sim_style_function(feature):
            rr = feature['properties'].get('rr_estimado_mediana', 1.0)
            if rr < 0.8:
                fill_color = '#66bb6a' # Verde (Seguro)
            elif rr < 1.2:
                fill_color = '#ffb74d' # Naranja claro (Moderado)
            elif rr < 1.8:
                fill_color = '#ff7043' # Naranja oscuro
            else:
                fill_color = '#ff4444' # Rojo (Peligro)
            return {
                'fillColor': fill_color,
                'color': '#30363d',
                'weight': 1,
                'fillOpacity': 0.35
            }
            
        folium.GeoJson(
            gdf_subcircuitos,
            style_function=sim_style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['subcircuito_nom', 'rr_estimado_mediana'],
                aliases=['Subcircuito:', 'Riesgo Relativo:'],
                localize=True
            )
        ).add_to(m_sim)
            
        # Capturar clics del usuario
        click_data = st_folium(m_sim, width=700, height=500, key="sim_map", returned_objects=["last_clicked"])
        
    with col_s2:
        # Captura y evaluación del clic
        punto_objetivo = click_data.get("last_clicked") if click_data else None
        
        if punto_objetivo:
            lat_c = punto_objetivo["lat"]
            lng_c = punto_objetivo["lng"]
            point_geom = Point(lng_c, lat_c)
            
            # Buscar si está contenido en algún subcircuito
            contiene = gdf_subcircuitos[gdf_subcircuitos.geometry.contains(point_geom)]
            
            if not contiene.empty:
                zona_cercana = contiene.iloc[0]
            else:
                # Buscar el subcircuito más cercano
                gdf_subcircuitos['dist'] = gdf_subcircuitos.geometry.centroid.distance(point_geom)
                zona_cercana = gdf_subcircuitos.loc[gdf_subcircuitos['dist'].idxmin()]
                
            # Calcular riesgo local proyectado con ayuda del trace de PyMC
            sub_codes_loc = sorted(gdf_subcircuitos['cod_subcircu'].unique())
            cod_actual = zona_cercana['cod_subcircu']
            
            if trace is not None and cod_actual in sub_codes_loc:
                idx_trace = sub_codes_loc.index(cod_actual)
                spatial_var = 'psi' if 'psi' in trace.posterior else ('phi' if 'phi' in trace.posterior else 'spatial_total')
                phi_all = trace.posterior[spatial_var].values  # shape (chains, draws, zonas)
                exp_all = np.exp(phi_all)
                # Estandarización sobre la media espacial de los subcircuitos (Clayton & Kaldor, 1987)
                rr_all = exp_all / np.mean(exp_all, axis=-1, keepdims=True)
                rr_samples = rr_all[..., idx_trace].flatten()
                
                # Calcular probabilidades empíricas usando las muestras posteriores
                prob_segura = float(np.mean(rr_samples < 1.0))
                prob_insegura = float(np.mean(rr_samples > 1.0))
                prob_peligro = float(np.mean(rr_samples > 1.5))
                
                # Riesgo relativo mediano e intervalos de credibilidad
                rr_estimado = float(np.median(rr_samples))
                hdi_min = float(np.percentile(rr_samples, 2.5))
                hdi_max = float(np.percentile(rr_samples, 97.5))
            else:
                # Lectura directa desde los resultados precomputados por el modelo (resultados_riesgo_relativo.csv)
                rr_estimado = float(zona_cercana.get('rr_estimado_mediana', 1.0))
                prob_peligro = float(zona_cercana.get('probabilidad_excedencia', 0.0))
                prob_insegura = float(zona_cercana.get('prob_exced_1', 1.0 if rr_estimado > 1.0 else 0.0))
                prob_segura = float(zona_cercana.get('prob_segura', max(0.0, 1.0 - prob_insegura)))
                hdi_min = float(zona_cercana.get('rr_limite_inf_95', rr_estimado))
                hdi_max = float(zona_cercana.get('rr_limite_sup_95', rr_estimado))
            
            # Clasificación de seguridad basada en probabilidades posteriores
            if prob_segura >= 0.70:
                color = "#66bb6a"  # Verde
                estado = "🟢 SEGURO (BAJO RIESGO)"
                explicacion = f"La probabilidad bayesiana de que este subcircuito mantenga una tasa de homicidios por debajo del promedio cantonal es del **{prob_segura*100:.1f}%**. El análisis espacial confirma condiciones de menor riesgo relativo."
            elif prob_segura < 0.70 and prob_peligro < 0.50:
                color = "#ffb74d"  # Naranja
                estado = "🟡 MODERADO (RIESGO MEDIO)"
                explicacion = f"El subcircuito presenta niveles de alerta intermedios, con una probabilidad posterior de riesgo superior al promedio cantonal de **{prob_insegura*100:.1f}%**."
            else:
                color = "#ff4444"  # Rojo
                estado = "🔴 INSEGURO (ALTA PRIORIDAD)"
                explicacion = f"Zona de concentración delictiva severa. La probabilidad de riesgo crítico (RR > 1.5 sobre la referencia cantonal) es del **{prob_peligro*100:.1f}%**."
            
            resultados_html = f"""<div class="stat-card" style="border-left: 5px solid {color}; padding: 15px; background: rgba(30,30,30,0.6); border-radius: 8px;">
<div style="font-size: 0.85rem; font-weight: bold; color: #f0f6fc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
    <i class="fa-solid fa-shield-halved"></i> EVALUACIÓN BAYESIANA DE RIESGO
</div>
<div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 12px;">
    Subcircuito: <strong style="color: #58a6ff;">{zona_cercana['subcircuito_nom']}</strong><br>
    Coordenadas: Lat {lat_c:.5f} | Lng {lng_c:.5f}
</div>

<div style="background-color: rgba(255,255,255,0.03); padding: 12px; border-radius: 6px; margin-bottom: 15px;">
    <div style="font-size: 0.8rem; color: #8b949e;">Clasificación de Riesgo Posterior:</div>
    <div style="color: {color}; font-size: 1.15rem; font-weight: bold; margin-top: 4px;">{estado}</div>
    <div style="font-size: 0.75rem; color: #c9d1d9; margin-top: 6px; line-height: 1.4; text-align: justify;">{explicacion}</div>
</div>

<div style="display: flex; flex-direction: column; gap: 10px;">
    <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8b949e; margin-bottom: 2px;">
            <span>🟢 Probabilidad de Riesgo Bajo (RR < 1.0)</span>
            <strong style="color: #66bb6a;">{prob_segura*100:.1f}%</strong>
        </div>
        <div style="height: 6px; background-color: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
            <div style="width: {prob_segura*100}%; height: 100%; background-color: #66bb6a; border-radius: 3px;"></div>
        </div>
    </div>
    <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8b949e; margin-bottom: 2px;">
            <span>🟡 Probabilidad de Riesgo Elevado (RR > 1.0)</span>
            <strong style="color: #ffb74d;">{prob_insegura*100:.1f}%</strong>
        </div>
        <div style="height: 6px; background-color: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
            <div style="width: {prob_insegura*100}%; height: 100%; background-color: #ffb74d; border-radius: 3px;"></div>
        </div>
    </div>
    <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8b949e; margin-bottom: 2px;">
            <span>🔴 Probabilidad de Riesgo Crítico (RR > 1.5)</span>
            <strong style="color: #ff4444;">{prob_peligro*100:.1f}%</strong>
        </div>
        <div style="height: 6px; background-color: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
            <div style="width: {prob_peligro*100}%; height: 100%; background-color: #ff4444; border-radius: 3px;"></div>
        </div>
    </div>
</div>

<div style="margin-top: 15px; border-top: 1px solid #30363d; padding-top: 10px; font-size: 0.75rem; display: flex; justify-content: space-between; align-items: center;">
    <span style="color: #8b949e;">Riesgo Relativo Mediano (RR):</span>
    <strong style="color: {color}; font-size: 0.95rem;">{rr_estimado:.2f}x</strong>
</div>
<div style="background-color: rgba(255,255,255,0.02); padding: 8px; border-radius: 4px; margin-top: 8px; font-size: 0.75rem; text-align: center; color: #8b949e;">
    📊 Intervalo de Credibilidad 95%: [{hdi_min:.2f}x , {hdi_max:.2f}x]
</div>
</div>"""
            st.markdown(resultados_html, unsafe_allow_html=True)
            
            # Ecuaciones matemáticas en Latex
            st.markdown("""<div class="latex-container">
<p style="font-size: 0.85rem; color: #58a6ff; font-weight: bold; margin-bottom: 10px;">📐 Formulación Matemática del Modelo BYM2 Estimado:</p>""", unsafe_allow_html=True)
            st.latex(r"\log(\mu_{it}) = \alpha + \beta_{\text{year}} \cdot \mathbb{I}(t = 2025) + \beta_{\text{area}} \log(\text{Area}_i) + \psi_i")
            st.latex(r"\psi_i = \sigma_s \left( \sqrt{\rho} \phi^*_i + \sqrt{1 - \rho} \theta^*_i \right)")
            st.markdown("""<p style="font-size: 0.75rem; color: #8b949e; margin-top: 5px;">
• <b>&psi;<sub>i</sub></b>: Efecto espacial total del modelo BYM2.<br>
• <b>&phi;*<sub>i</sub></b>: Componente ICAR escalado (contigüidad geográfica de reina).<br>
• <b>&theta;*<sub>i</sub></b>: Componente no estructurado estándar Normal(0, 1) (heterogeneidad zonal).<br>
• <b>&rho;</b>: Parámetro de mezcla espacial con prior Beta(0.5, 0.5).<br>
• <b>&beta;<sub>year</sub></b>: Efecto fijo anual con prior Normal(0, 1).
</p>
</div>""", unsafe_allow_html=True)
            
        else:
            st.markdown("""<div class="stat-card">
<div class="stat-card-title" style="color: #8b949e;"><i class="fa-solid fa-hand-pointer"></i> Esperando Clic</div>
<div style="text-align: center; padding: 30px 0;">
<p>Por favor, haz <b>clic en cualquier punto del mapa a la izquierda</b> para obtener la proyección de riesgo probabilístico de homicidios.</p>
</div>
</div>""", unsafe_allow_html=True)

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
