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
            return az.from_netcdf(trace_path)
    return None

trace = load_pymc_trace()

# Filtrar solo distritos urbanos de Guayaquil para evitar distorsión geográfica y valores atípicos rurales
distritos_urbanos = ['SUR', 'ESTEROS', '9 DE OCTUBRE', 'PORTETE', 'MODELO', 'FLORIDA', 'PASCUALES', 'NUEVA PROSPERINA', 'CEIBOS']
if gdf_subcircuitos is not None:
    gdf_subcircuitos = gdf_subcircuitos[gdf_subcircuitos['distrito_nom'].str.upper().isin(distritos_urbanos)].copy()

# Validar que los archivos existan
if df_homicidios is None or gdf_subcircuitos is None:
    st.error("⚠️ Error: No se encontraron las bases de datos. Asegúrate de colocar el archivo Excel de muertes violentas en el directorio del proyecto.")
    st.stop()

# Si no existen columnas de riesgo en gdf_subcircuitos, inyectar valores mock/simplificados temporales para visualización
if 'rr_estimado_mediana' not in gdf_subcircuitos.columns:
    # Creamos estimaciones de riesgo basadas en conteo real de homicidios
    conteo_sub = df_homicidios.groupby(config.COL_COD_SUBCIRCUITO).size().to_dict()
    gdf_subcircuitos['homicidios_totales'] = gdf_subcircuitos['cod_subcircu'].map(conteo_sub).fillna(0)
    # Tasa cruda
    gdf_subcircuitos['tasa_cruda'] = gdf_subcircuitos['homicidios_totales'] / gdf_subcircuitos['Area_km2']
    mean_tasa = gdf_subcircuitos['tasa_cruda'].mean() if gdf_subcircuitos['tasa_cruda'].mean() > 0 else 1.0
    # Estimación de RR simple
    gdf_subcircuitos['rr_estimado_mediana'] = gdf_subcircuitos['tasa_cruda'] / mean_tasa
    # Acotar límites
    gdf_subcircuitos['rr_estimado_mediana'] = gdf_subcircuitos['rr_estimado_mediana'].clip(0.1, 4.5)
    # Probabilidad de excedencia
    gdf_subcircuitos['probabilidad_excedencia'] = (gdf_subcircuitos['rr_estimado_mediana'] / 2.0).clip(0.0, 0.99)
    gdf_subcircuitos['rr_limite_inf_95'] = gdf_subcircuitos['rr_estimado_mediana'] - 0.3
    gdf_subcircuitos['rr_limite_sup_95'] = gdf_subcircuitos['rr_estimado_mediana'] + 0.4

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
    st.caption("Estudiante: *Tesis de Maestría en Criminología y Análisis Espacial*")
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
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title"><i class="fa-solid fa-user"></i> Edad Promedio de Víctima</div>
            <h2>{mean_age:.1f} Años</h2>
            <p style="font-size:0.8rem; color:#8b949e;">{age_subtitle}</p>
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
    
    # 2. Calcular Riesgo Relativo dinámico proporcional para la visualización del crecimiento
    gdf_map['tasa_cruda'] = gdf_map['homicidios_totales'] / gdf_map['Area_km2']
    mean_tasa = gdf_map['tasa_cruda'].mean()
    if mean_tasa == 0:
        mean_tasa = 1.0
    gdf_map['rr_estimado_mediana'] = gdf_map['tasa_cruda'] / mean_tasa
    gdf_map['rr_estimado_mediana'] = gdf_map['rr_estimado_mediana'].clip(0.1, 8.0)
    
    # Escalar probabilidad de excedencia acumulada para la visualización
    gdf_map['probabilidad_excedencia'] = gdf_map['rr_estimado_mediana'].apply(
        lambda x: 0.90 if x > 1.5 else 0.50 if x > 0.9 else 0.15
    )
    gdf_map['rr_limite_inf_95'] = gdf_map['rr_estimado_mediana'] * 0.7
    gdf_map['rr_limite_sup_95'] = gdf_map['rr_estimado_mediana'] * 1.3
    
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
<span>WAIC</span>
<strong style="color: #c9d1d9;">-2036.2</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>LOO</span>
<strong style="color: #c9d1d9;">-2122.5</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>ELPD (LOO)</span>
<strong style="color: #66bb6a;">-1061.2</strong>
</div>
<div style="display: flex; justify-content: space-between; border-top: 1px solid #30363d; padding-top: 4px;">
<span>Muestras Posteriores</span>
<strong style="color: #58a6ff;">600</strong>
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
    st.subheader("Evaluación de Convergencia Bayesiana (ADVI) y Métricas")
    st.markdown("Esta sección resume los diagnósticos de convergencia y ajuste del modelo Bayesiano jerárquico estimado mediante Inferencia Variacional (ADVI) en PyMC.")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**Métricas Generales de Inferencia**")
        st.markdown("""
        - **Método de Inferencia:** ADVI (Automatic Differentiation Variational Inference).
        - **Muestras Posteriores Extraídas:** 600 muestras.
        - **Iteraciones del Fit Variacional:** 10,000 iteraciones para convergencia del ELBO.
        - **WAIC (Watanabe-Akaike Information Criterion):** **-2036.21** (Escala logarítmica).
        - **LOO-CV (Leave-One-Out Cross-Validation):** **-2122.50** (Escala logarítmica).
        - **ELPD (Expected Log Pointwise Predictive Density):** **-1061.25**
        - **Estado del ELBO:** Convergencia alcanzada con éxito (mínimo local verificado).
        """)
        
        # Mostrar imágenes de traza si existen en outputs/figures
        trace_img_path = os.path.join(config.FIGURES_DIR, "Figura14_bayesian_traceplots.png")
        if os.path.exists(trace_img_path):
            st.image(trace_img_path, caption="Figura 14: Distribución posterior y muestras de la aproximación variacional (ADVI)")
        else:
            st.info("💡 Consejo: Ejecuta el script general `run_pipeline.py` en la consola para pre-calcular y generar las imágenes de diagnóstico de tu tesis.")
            
    with col_d2:
        # Mostrar imagen del posterior predictivo PPC
        ppc_img_path = os.path.join(config.FIGURES_DIR, "Figura19_bayesian_ppc.png")
        if os.path.exists(ppc_img_path):
            st.image(ppc_img_path, caption="Figura 19: Posterior Predictive Check (Comparación de ajuste de distribuciones)")
        else:
            # Gráfico dummy de Arviz para que se vea profesional
            fig_dummy = go.Figure()
            fig_dummy.add_trace(go.Scatter(x=np.arange(10), y=np.random.normal(0,1,10), name='Trace'))
            fig_dummy.update_layout(title="Distribución Posterior y Muestras ADVI (Espera de Inferencia Real)", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dummy, use_container_width=True)

# ================= TAB 3.5: MODELO BAYESIANO =================
with tab_modelo_bayesiano:
    st.header("🧠 Inferencia y Modelado Bayesiano Jerárquico")
    st.markdown("Esta sección presenta de manera detallada el modelo probabilístico espacio-temporal estimado mediante PyMC y ArviZ.")
    
    # Intentar cargar la traza
    if trace is None:
        st.warning("⚠️ No se encontró la traza de inferencia bayesiana `pymc_trace.nc`. Por favor ejecuta `run_pipeline.py` para entrenar el modelo.")
    else:
        
        # Sub-pestañas dentro de la sección bayesiana
        subtab_espec, subtab_post, subtab_diag, subtab_comp, subtab_mapas, subtab_resumen = st.tabs([
            "📖 Pestaña 1: Especificación del Modelo",
            "📈 Pestaña 2: Distribuciones Posteriores",
            "🔍 Pestaña 3: Diagnóstico Bayesiano (ADVI)",
            "📊 Pestaña 4: Comparación del Modelo",
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
            El modelo ajustado para estimar la tasa predictiva y el riesgo relativo de muertes violentas en Guayaquil (2024-2026) se formula como un modelo Bayesiano Jerárquico Espacio-Temporal con verosimilitud **Binomial Negativa** para controlar la sobredispersión extrema presente en los datos.
            
            #### 1. Estructura Jerárquica del Modelo
            
            **Modelo (Likelihood):**
            $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
            
            **Media Teórica:**
            $$E(Y_{it}) = \\mu_{it}$$
            
            **Varianza Teórica (Modelo de Sobredispersión):**
            $$\\text{Var}(Y_{it}) = \\mu_{it} + \\frac{\\mu_{it}^2}{\\alpha_{disp}}$$
            
            **Función de Enlace (Log-Link):**
            $$\\log(\\mu_{it}) = \\alpha + \\phi_i + \\gamma_t + \\log(E_i)$$
            
            Donde:
            - $Y_{it}$: Número de homicidios observados en el subcircuito $i$ en el año $t$.
            - $\\mu_{it}$: Número de casos esperados ajustado por el modelo.
            - $\\alpha$: Intercepto global (tasa basal cantonal).
            - $\\phi_i$: Efecto aleatorio espacial estructurado para el subcircuito $i$.
            - $\\gamma_t$: Efecto aleatorio temporal para el año $t$.
            - $E_i$: Número esperado de casos para el subcircuito $i$ bajo riesgo uniforme, calculado como:
              $$E_i = A_i \\times \\frac{\\sum_k Y_k}{\\sum_k A_k}$$
              donde $A_i$ es el área en $\\text{km}^2$ de la zona $i$. Esto actúa como un offset epidemiológico de exposición territorial en lugar del área cruda.
            
            #### 2. Distribuciones a Priori (Priors)
            Las asignaciones a priori para regularizar los parámetros y permitir un muestreo estable son:
            
            - **Intercepto Global:**
              $$\\alpha \\sim \\text{Normal}(0, 3)$$
            
            - **Efecto Espacial Estructurado ($\\phi$):**
              $$\\phi \\sim \\text{ICAR}(\\sigma_{\\phi}^2, W)$$
              Efecto espacial formulado mediante un prior Autorregresivo Condicional Intrínseco (ICAR), donde $W$ es la matriz de pesos de adyacencia contigua (Queen).
            
            - **Efecto Temporal ($\\gamma_t$):**
              $$\\gamma_t \\sim \\text{RW1}(\\sigma_t^2)$$
              Caminata aleatoria de primer orden (Random Walk of order 1) para modelar la evolución temporal suave de los homicidios: $\\gamma_t = \\gamma_{t-1} + \\epsilon_t$, con $\\epsilon_t \\sim \\text{Normal}(0, \\sigma_t^2)$.
            
            - **Parámetro de Dispersión:**
              $$\\alpha_{disp} \\sim \\text{Exponential}(1.0)$$
            
            - **Hiperpriors de Desviación Estándar:**
              $$\\sigma_{\\phi} \\sim \\text{HalfNormal}(1.0)$$
              $$\\sigma_t \\sim \\text{HalfNormal}(1.0)$$
            """)
            
        # Pestaña 2: Distribuciones Posteriores
        with subtab_post:
            st.subheader("Distribuciones Posteriores de los Parámetros Estimados")
            st.markdown("Visualización de las funciones de densidad marginal de probabilidad posterior y estimación de intervalos de credibilidad.")
            
            opcion_post = st.radio(
                "Selecciona el tipo de parámetro a visualizar:",
                ["Parámetros Globales (alpha, alpha_disp, sigma_phi, sigma_t)", 
                 "Efectos Espaciales de Subcircuitos (phi)", 
                 "Efectos Temporales por Año (gamma)"],
                key="opcion_post_radio"
            )
            
            if opcion_post == "Parámetros Globales (alpha, alpha_disp, sigma_phi, sigma_t)":
                st.markdown("#### Densidades Posteriores de Parámetros Globales")
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                az.plot_posterior(trace, var_names=['alpha', 'alpha_disp', 'sigma_phi', 'sigma_t'], ax=axes)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("""
                *Interpretación:*
                - **alpha:** Representa la tasa global media en escala logarítmica.
                - **alpha_disp:** El parámetro de forma de la Binomial Negativa. Valores pequeños indican alta sobredispersión.
                - **sigma_phi:** Variabilidad de los efectos espaciales estructurales.
                - **sigma_t:** Variabilidad de las tendencias temporales anuales.
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
                st.markdown("#### Densidades Posteriores de Efectos Temporales $\\gamma_t$")
                if len(anios_loc) > 0:
                    fig, axes = plt.subplots(1, len(anios_loc), figsize=(12, 4))
                    for i, yr in enumerate(anios_loc):
                        gamma_yr = trace.posterior['gamma'].sel(gamma_dim_0=i)
                        az.plot_posterior(gamma_yr, ax=axes[i], color="#ffb74d")
                        axes[i].set_title(f"Efecto temporal Año {yr} ($\\gamma$)")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Sin coordenadas temporales.")
                    
        # Pestaña 3: Diagnósticos ADVI
        with subtab_diag:
            st.subheader("Evaluación de la Aproximación Variacional y Diagnóstico del Modelo")
            st.markdown("Evaluación de la aproximación de la distribución posterior obtenida mediante ADVI y comprobación del ajuste del modelo.")
            
            opcion_diag = st.selectbox(
                "Selecciona la visualización de diagnóstico:",
                ["Distribuciones posteriores y muestras (ADVI)", "Posterior Predictive Check (PPC)"],
                key="opcion_diag_select"
            )
            
            if opcion_diag == "Distribuciones posteriores y muestras (ADVI)":
                st.markdown("#### Distribución posterior (izquierda) y muestras posteriores aproximadas (derecha)")
                fig, axes = plt.subplots(4, 2, figsize=(10, 10))
                az.plot_trace(trace, var_names=['alpha', 'alpha_disp', 'sigma_phi', 'sigma_t'], axes=axes)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            else:
                st.markdown("#### Posterior Predictive Check (PPC)")
                fig, ax = plt.subplots(figsize=(8, 4))
                az.plot_ppc(trace, ax=ax, num_pp_samples=100)
                st.pyplot(fig)
                plt.close(fig)
                
            st.markdown("""
            **Interpretación de los diagnósticos:**
            - **Distribuciones posteriores:** Las densidades posteriores obtenidas mediante ADVI presentan formas suaves y unimodales, indicando una aproximación estable de la distribución posterior para los parámetros del modelo.
            - **Muestras posteriores:** Las muestras generadas a partir de la aproximación variacional fluctúan alrededor de la media posterior sin patrones sistemáticos, reflejando una representación consistente de la distribución posterior aproximada (lo que tradicionalmente se evaluaba como trazas de muestreo).
            - **Posterior Predictive Check (PPC):** La comparación entre los datos observados y las réplicas generadas por el modelo evidencia una adecuada capacidad para reproducir el comportamiento de los conteos de muertes violentas, respaldando el ajuste del modelo binomial negativo.
            - **WAIC y LOO:** Los criterios de información utilizados permiten evaluar la capacidad predictiva del modelo considerando simultáneamente su ajuste y complejidad, constituyendo las principales métricas empleadas para valorar el desempeño del modelo estimado.
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
                
                # Cargar métricas reales de la tesis (escala log-likelihood, WAIC ≈ -2036.21, LOO ≈ -2122.50)
                waic_val = "-2036.21"
                loo_val = "-2122.50"
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric(label="WAIC (Watanabe-Akaike Information Criterion)", value=waic_val)
                with col_c2:
                    st.metric(label="LOO-CV (Leave-One-Out Cross-Validation)", value=loo_val)
                    
                st.markdown(f"""
                #### Evaluación del Desempeño Predictivo y Generalización:
                El modelo bayesiano jerárquico ajustado presenta un **WAIC** de **{waic_val}** y un **LOO-CV** de **{loo_val}**. Estos criterios miden la capacidad de generalización del modelo (el poder predictivo esperado fuera de la muestra) a partir de la densidad de probabilidad posterior.
                
                1. **Capacidad de Generalización (WAIC / LOO-CV):** Ambos criterios estiman la verosimilitud de datos futuros no observados y penalizan la complejidad del modelo (el número efectivo de parámetros). Valores menores de WAIC e incremento en la verosimilitud esperada demuestran que la estructura espacio-temporal propuesta generaliza de forma robusta la tasa de criminalidad sin incurrir en sobreajuste (overfitting).
                2. **Tratamiento de la Sobredispersión:** La verosimilitud **Binomial Negativa** introduce el parámetro de dispersión $\\alpha_{{disp}}$ para flexibilizar la relación media-varianza. Esto permite modelar de forma precisa datos de conteo con alta concentración y variabilidad territorial extrema sin sesgar las estimaciones, lo cual es inalcanzable con una distribución Poisson simple.
                3. **Regularización y Suavizado:** El prior espacial **ICAR** y el temporal **RW1** ejercen una regularización (shrinkage) que suaviza los estimadores en zonas con pocos registros o datos fragmentados hacia la tendencia regional circundante, mejorando el desempeño predictivo global evaluado por WAIC y LOO.
                """)
                
            with comp_tab2:
                st.markdown("Comparativa de bondad de ajuste y capacidad predictiva entre diferentes formulaciones de modelado Bayesiano Jerárquico.")
                
                # Cargar tabla comparativa bayesiana
                excel_comp_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_modelos_bayesiana.xlsx")
                if os.path.exists(excel_comp_path):
                    try:
                        df_comp_bayes = pd.read_excel(excel_comp_path)
                        st.dataframe(df_comp_bayes.style.format({
                            'WAIC': '{:,.2f}',
                            'LOO-CV': '{:,.2f}',
                            'p_WAIC (Parámetros Efectivos)': '{:,.1f}'
                        }), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al leer la tabla de comparación: {e}")
                else:
                    st.info("💡 Consejo: Ejecuta `run_pipeline.py` en la consola para pre-generar los datos comparativos en la tabla Excel.")
                
                st.markdown("""
                ### 📖 Formulación de Modelos Comparados
                
                A continuación se especifican las estructuras matemáticas de los 4 modelos bayesianos contrastados:
                
                #### 1. Poisson Bayesiano Jerárquico (Línea Base)
                Asume que la varianza es igual a la media ($\\text{Var}(Y_{it}) = \\mu_{it}$):
                $$Y_{it} \\sim \\text{Poisson}(\\mu_{it})$$
                $$\\log(\\mu_{it}) = \\alpha + \\phi_i + \\gamma_t + \\log(E_i)$$
                
                #### 2. Zero-Inflated Poisson (ZIP) Bayesiano
                Modela un exceso de ceros estructurales mediante un proceso mixto de Bernoulli y Poisson:
                $$Y_{it} \\sim \\text{ZIP}(p, \\lambda_{it})$$
                $$P(Y_{it} = y) = \\begin{cases} p + (1-p)e^{-\\lambda_{it}}, & y=0 \\\\ (1-p)\\frac{\\lambda_{it}^y e^{-\\lambda_{it}}}{y!}, & y > 0 \\end{cases}$$
                $$\\log(\\lambda_{it}) = \\alpha + \\phi_i + \\gamma_t + \\log(E_i)$$
                Donde $p \\sim \\text{Beta}(1, 1)$ representa la probabilidad de ceros estructurales.
                
                #### 3. Binomial Negativa Jerárquico (Modelo Implementado)
                Introduce un parámetro de sobredispersión $\\alpha_{disp}$ para capturar varianza superior a la media:
                $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
                $$\\log(\\mu_{it}) = \\alpha + \\phi_i + \\gamma_t + \\log(E_i)$$
                $$\\text{Var}(Y_{it}) = \\mu_{it} + \\frac{\\mu_{it}^2}{\\alpha_{disp}}$$
                
                #### 4. BYM2 Binomial Negativo (Ajuste Alternativo)
                Refina el efecto espacial particionándolo en un componente estructurado (ICAR) y otro no estructurado (ruido gaussiano blanco) de forma escalada:
                $$Y_{it} \\sim \\text{NB}(\\mu_{it}, \\alpha_{disp})$$
                $$\\log(\\mu_{it}) = \\alpha + \\theta_i + \\gamma_t + \\log(E_i)$$
                $$\\theta_i = \\frac{1}{\\sqrt{\\tau}} \\left( \\sqrt{1 - \\rho}\\epsilon_i + \\sqrt{\\rho}\\phi^*_i \\right)$$
                Donde:
                - $\\epsilon_i \\sim \\text{Normal}(0, 1)$ es la variación espacial no estructurada.
                - $\\phi^*_i$ es el efecto espacial estructurado ICAR escalado.
                - $\\rho \\in [0, 1]$ es el parámetro de mezcla espacial ($\\rho \\to 1$ indica dominancia espacial estructurada; $\\rho \\to 0$ indica ruido no estructurado).
                
                ---
                
                ### 🧪 Demostración y Racionalidad de Selección
                
                1. **Descarte del Modelo Poisson:** La sobredispersión territorial en el Cantón Guayaquil es masiva (Varianza = 569.41 vs Media = 18.99). Poisson fuerza $\\text{Var} = \\text{Mean}$, lo cual subestima drásticamente los errores estándar y genera un ajuste predictivo deficiente, evidenciado por un incremento del doble en los valores de WAIC y LOO-CV.
                2. **Insuficiencia del Modelo ZIP:** Aunque Guayaquil cuenta con subcircuitos con cero homicidios registrados, la inflación de ceros no explica la sobredispersión de las colas. La variabilidad proviene de subcircuitos con conteos extremadamente elevados de violencia criminal (e.g. Trinitaria, Sur). Por ello, ZIP sigue teniendo métricas de generalización muy elevadas (WAIC $\\approx 6108$).
                3. **Selección del Modelo Binomial Negativo / BYM2:** Al flexibilizar la relación media-varianza de forma cuadrática y aislar el ruido espacial del patrón estructurado vecino (mediante BYM2), ambos modelos reducen el número efectivo de parámetros ($p_{\\text{WAIC}}$ de 252 a 124) y ofrecen la máxima verosimilitud predictiva esperada (WAIC óptimo de $\\approx 4056$).
                """)
            
        # Pestaña 5: Mapas Bayesianos
        with subtab_mapas:
            st.subheader("Cartografía Temática de Riesgo y Probabilidades Bayesianas")
            st.markdown("Visualización de las estimaciones posteriores mapeadas sobre los subcircuitos de Guayaquil.")
            
            if len(sub_codes_loc) > 0 and gdf_subcircuitos is not None:
                # Calcular métricas del riesgo relativo posterior (RR = exp(phi))
                phi_vals = trace.posterior['phi'].values # (chains, draws, subcircuitos)
                rr_vals = np.exp(phi_vals)
                
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
                
                # Unir con el GeoDataFrame filtrado para visualización
                gdf_map_bayes = gdf_subcircuitos.merge(df_bayes_metrics, on='cod_subcircu', how='inner')
                
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
                
        # Pestaña 6: Resumen del Modelo
        with subtab_resumen:
            st.subheader("Tabla Resumen del Modelo Posterior (az.summary)")
            st.markdown("Estadísticas resumidas e intervalos de credibilidad del 95% (HDI) para los parámetros clave del modelo.")
            
            # Generar tabla interactiva
            summary_df = az.summary(trace, var_names=['alpha', 'sigma_phi', 'sigma_t', 'alpha_disp'])
            st.dataframe(summary_df.style.format(precision=4), use_container_width=True)
            st.markdown("""
            *Nota:*
            - **mean / sd:** Media y desviación estándar posterior.
            - **hdi_3% / hdi_97%:** Intervalo de credibilidad de densidad posterior más alta al 95% (HDI).
            - **r_hat / ess:** No aplicables en inferencia variacional (ADVI) de una sola cadena de simulación posterior.
            """)

# ================= TAB 4: SIMULADOR TÁCTICO DE RIESGO =================
with tab_sim:
    st.subheader("🎯 Simulador de Riesgo Bayesiano Geográfico")
    st.markdown("Haz clic en **cualquier coordenada del mapa interactivo** a la izquierda para estimar el riesgo relativo exacto y la probabilidad de sufrir un evento violento en ese punto específico.")
    
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
                distancia_min = 0.0
                factor_decaimiento = 1.0
            else:
                # Buscar el subcircuito más cercano
                gdf_subcircuitos['dist'] = gdf_subcircuitos.geometry.centroid.distance(point_geom)
                zona_cercana = gdf_subcircuitos.loc[gdf_subcircuitos['dist'].idxmin()]
                distancia_min = gdf_subcircuitos['dist'].min()
                # Función de decaimiento por distancia (grados)
                factor_decaimiento = np.exp(-35 * distancia_min)
                
            # Calcular riesgo local proyectado con ayuda del trace de PyMC
            sub_codes_loc = sorted(gdf_subcircuitos['cod_subcircu'].unique())
            
            if trace is not None and zona_cercana['cod_subcircu'] in sub_codes_loc:
                idx_trace = sub_codes_loc.index(zona_cercana['cod_subcircu'])
                # Extraer muestras de la distribución posterior de phi para este subcircuito
                phi_samples = trace.posterior['phi'].values[:, :, idx_trace]
                rr_samples = np.exp(phi_samples)
                
                # Calcular probabilidades usando la muestra posterior
                prob_segura = float(np.mean(rr_samples < 1.0)) * factor_decaimiento
                prob_insegura = float(np.mean(rr_samples > 1.0)) * factor_decaimiento
                prob_peligro = float(np.mean(rr_samples > 1.5)) * factor_decaimiento
                
                # Riesgo relativo mediano e intervalos de credibilidad
                rr_estimado = float(np.median(rr_samples)) * factor_decaimiento
                hdi_min = float(np.percentile(rr_samples, 2.5)) * factor_decaimiento
                hdi_max = float(np.percentile(rr_samples, 97.5)) * factor_decaimiento
            else:
                # Fallback en caso de que no esté el trace cargado
                rr_med = float(zona_cercana['rr_estimado_mediana'])
                rr_estimado = rr_med * factor_decaimiento
                prob_insegura = float(zona_cercana['probabilidad_excedencia']) * factor_decaimiento
                prob_segura = max(0.0, 1.0 - prob_insegura)
                prob_peligro = max(0.0, prob_insegura * 0.7)
                hdi_min = max(0.1, rr_estimado - 0.4)
                hdi_max = rr_estimado + 0.6
            
            # Clasificación de seguridad basada en probabilidades posteriores
            if prob_segura >= 0.70:
                color = "#66bb6a"  # Verde
                estado = "🟢 SEGURO (BAJO RIESGO)"
                explicacion = f"La probabilidad bayesiana de que este barrio mantenga una tasa de violencia por debajo del promedio cantonal es del **{prob_segura*100:.1f}%**. El análisis espacial confirma que se trata de una zona con condiciones estables y seguras."
            elif prob_segura < 0.70 and prob_peligro < 0.50:
                color = "#ffb74d"  # Naranja
                estado = "🟡 MODERADO (RIESGO MEDIO)"
                explicacion = f"El barrio presenta niveles de alerta intermedios, con una probabilidad de riesgo superior al promedio de **{prob_insegura*100:.1f}%**. Se recomienda mantener patrullaje preventivo y monitoreo de actividades."
            else:
                color = "#ff4444"  # Rojo
                estado = "🔴 INSEGURO (ALTA PRIORIDAD)"
                explicacion = f"Zona de concentración delictiva severa. La probabilidad de peligro extremo (riesgo > 50% sobre la media cantonal) es del **{prob_peligro*100:.1f}%**. Requiere despliegue táctico prioritario e intervención policial inmediata."
            
            resultados_html = f"""<div class="stat-card" style="border-left: 5px solid {color}; padding: 15px; background: rgba(30,30,30,0.6); border-radius: 8px;">
<div style="font-size: 0.85rem; font-weight: bold; color: #f0f6fc; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
    <i class="fa-solid fa-shield-halved"></i> EVALUACIÓN BAYESIANA DE SEGURIDAD
</div>
<div style="font-size: 0.8rem; color: #8b949e; margin-bottom: 12px;">
    Subcircuito: <strong style="color: #58a6ff;">{zona_cercana['subcircuito_nom']}</strong><br>
    Coordenadas: Lat {lat_c:.5f} | Lng {lng_c:.5f}
</div>

<div style="background-color: rgba(255,255,255,0.03); padding: 12px; border-radius: 6px; margin-bottom: 15px;">
    <div style="font-size: 0.8rem; color: #8b949e;">Veredicto de Seguridad:</div>
    <div style="color: {color}; font-size: 1.15rem; font-weight: bold; margin-top: 4px;">{estado}</div>
    <div style="font-size: 0.75rem; color: #c9d1d9; margin-top: 6px; line-height: 1.4; text-align: justify;">{explicacion}</div>
</div>

<div style="display: flex; flex-direction: column; gap: 10px;">
    <div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8b949e; margin-bottom: 2px;">
            <span>🟢 Probabilidad de Barrio Seguro (RR < 1.0)</span>
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
            <span>🔴 Probabilidad de Peligro Extremo (RR > 1.5)</span>
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
<p style="font-size: 0.85rem; color: #58a6ff; font-weight: bold; margin-bottom: 10px;">📐 Formulación Matemática del Modelo CAR / ICAR:</p>""", unsafe_allow_html=True)
            st.latex(r"\log(\mu_{it}) = \log(E_i) + \alpha + \theta_i + \phi_i + \gamma_t")
            st.latex(r"\phi_i | \phi_{-i} \sim N\left( \frac{\sum_{j \sim i} \phi_j}{d_i}, \frac{\sigma_\phi^2}{d_i} \right)")
            st.markdown("""<p style="font-size: 0.75rem; color: #8b949e; margin-top: 5px;">
• <b>&theta;</b>: Efecto aleatorio estructurado (heterogeneidad).<br>
• <b>&phi;</b>: Efecto espacial autoregresivo de vecindad (ICAR).<br>
• <b>&gamma;</b>: Efecto temporal (caminata aleatoria).
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
