import pandas as pd
import geopandas as gpd
import numpy as np
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage3")

def run_stage3(df: pd.DataFrame) -> tuple:
    """
    Etapa 3: Ingeniería de Variables y Agregación Espacio-Temporal
    - Agrupa edad en rangos lógicos.
    - Agrupa tipo de arma en categorías unificadas.
    - Carga el shapefile de subcircuitos de Guayaquil.
    - Une los eventos con las geometrías oficiales.
    - Calcula el conteo mensual y anual de homicidios por subcircuito.
    - Calcula tasas de delincuencia espacial (crimes/km²).
    - Genera base de modelado y guarda los resultados agregados.
    """
    logger.info("Iniciando ETAPA 3: Ingeniería de Variables y Agregación")
    
    # 1. Agrupamiento de Edad
    logger.info("Agrupando edades...")
    df['Edad_Limpia'] = pd.to_numeric(df[config.COL_EDAD], errors='coerce').fillna(30)
    bins = [0, 18, 30, 50, 120]
    labels = ['0-17 ANOS', '18-29 ANOS', '30-49 ANOS', '50+ ANOS']
    df['Grupo_Edad'] = pd.cut(df['Edad_Limpia'], bins=bins, labels=labels, right=False)
    
    # 2. Agrupamiento de Armas
    logger.info("Agrupando tipos de armas...")
    def clasificar_arma(arma_str):
        if not isinstance(arma_str, str):
            return 'OTROS'
        arma_str = arma_str.upper()
        if 'FUEGO' in arma_str or 'PISTOLA' in arma_str or 'REV' in arma_str or 'FUSIL' in arma_str:
            return 'ARMA DE FUEGO'
        elif 'BLANCA' in arma_str or 'CUCHILLO' in arma_str or 'MACHETE' in arma_str or 'PUNZ' in arma_str:
            return 'ARMA BLANCA'
        elif 'CONTUNDENTE' in arma_str or 'PALO' in arma_str or 'PIEDRA' in arma_str or 'FIERRO' in arma_str:
            return 'OBJETO CONTUNDENTE'
        elif 'ASFIXIA' in arma_str or 'ESTRANG' in arma_str or 'AHORC' in arma_str:
            return 'ASFIXIA'
        else:
            return 'OTROS'
            
    df['Grupo_Arma'] = df[config.COL_ARMA].apply(clasificar_arma)
    
    # 3. Agrupamiento de Tipo de Muerte (Sicariato, Asesinato, Homicidio, Femicidio)
    def clasificar_muerte(muerte_str):
        if not isinstance(muerte_str, str):
            return 'HOMICIDIO'
        muerte_str = muerte_str.upper()
        if 'ASESINATO' in muerte_str:
            return 'ASESINATO'
        elif 'SICARIATO' in muerte_str:
            return 'SICARIATO'
        elif 'FEMICIDIO' in muerte_str:
            return 'FEMICIDIO'
        else:
            return 'HOMICIDIO'
            
    df['Tipo_Muerte'] = df[config.COL_TIPO_MUERTE].apply(clasificar_muerte)

    # 4. Cargar Shapefile de Subcircuitos de Guayaquil
    logger.info(f"Cargando shapefile desde: {config.SHAPEFILE_PATH}")
    if not os.path.exists(config.SHAPEFILE_PATH):
        raise FileNotFoundError(f"No se encontró el shapefile en {config.SHAPEFILE_PATH}")
        
    gdf_sub = gpd.read_file(config.SHAPEFILE_PATH)
    # Filtrar solo subcircuitos correspondientes a Guayaquil
    gdf_sub = gdf_sub[gdf_sub['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
    
    # Asegurar que el CRS de la geometría esté definido
    if gdf_sub.crs is None:
        gdf_sub = gdf_sub.set_crs("EPSG:32717")
        
    # Disolver para agrupar geometrías multiparte bajo el mismo código único de subcircuito
    logger.info("Disolviendo shapefile por COD_SUBCIR para unir multipoligonos...")
    gdf_sub = gdf_sub.dissolve(by='COD_SUBCIR', as_index=False, aggfunc='first')
    
    # Calcular Área en Kilómetros Cuadrados directamente a partir de la geometría UTM disuelta
    gdf_sub['Area_km2'] = gdf_sub.geometry.area / 1e6
    
    # Renombrar columnas geográficas en el shapefile para facilidad
    gdf_sub = gdf_sub.rename(columns={
        'COD_SUBCIR': 'cod_subcircu',
        'NAM_SUBCIR': 'subcircuito_nom',
        'NAM_CIRCUI': 'circuito_nom',
        'NAM_DISTRI': 'distrito_nom'
    })
    
    # 5. Agregación espacio-temporal por subcircuito
    logger.info("Agregando homicidios por subcircuito y tiempo...")
    
    # Agregación Anual
    crime_anual = df.groupby([config.COL_COD_SUBCIRCUITO, 'Anio']).size().reset_index(name='homicidios')
    crime_anual.columns = ['cod_subcircu', 'Anio', 'homicidios']
    
    # Agregación Mensual
    crime_mensual = df.groupby([config.COL_COD_SUBCIRCUITO, 'Anio', 'Mes_Num', 'Mes']).size().reset_index(name='homicidios')
    crime_mensual.columns = ['cod_subcircu', 'Anio', 'Mes_Num', 'Mes', 'homicidios']
    
    # 6. Combinar con el Shapefile
    # Generamos una cuadrícula de todos los subcircuitos x todos los años para evitar omitir zonas con 0 casos
    sub_codes = gdf_sub['cod_subcircu'].unique()
    anios_unicos = df['Anio'].unique()
    
    grid_anual = pd.MultiIndex.from_product([sub_codes, anios_unicos], names=['cod_subcircu', 'Anio']).to_frame().reset_index(drop=True)
    df_anual_completo = grid_anual.merge(crime_anual, on=['cod_subcircu', 'Anio'], how='left').fillna({'homicidios': 0})
    
    # Agregar información geográfica al conteo anual
    gdf_anual = gdf_sub[['cod_subcircu', 'subcircuito_nom', 'circuito_nom', 'distrito_nom', 'Area_km2', 'geometry']].merge(
        df_anual_completo, on='cod_subcircu', how='right'
    )
    
    # Calcular Tasas (Homicidios por km² por año)
    gdf_anual['Tasa_Homicidios_km2'] = gdf_anual['homicidios'] / gdf_anual['Area_km2']
    
    # Agregación Mensual Completa
    meses_num = sorted(df['Mes_Num'].unique())
    grid_mensual = pd.MultiIndex.from_product([sub_codes, anios_unicos, meses_num], names=['cod_subcircu', 'Anio', 'Mes_Num']).to_frame().reset_index(drop=True)
    # Añadir el nombre del mes
    mes_num_to_name = df[['Mes_Num', 'Mes']].drop_duplicates().set_index('Mes_Num')['Mes'].to_dict()
    grid_mensual['Mes'] = grid_mensual['Mes_Num'].map(mes_num_to_name)
    
    df_mensual_completo = grid_mensual.merge(crime_mensual, on=['cod_subcircu', 'Anio', 'Mes_Num', 'Mes'], how='left').fillna({'homicidios': 0})
    gdf_mensual = gdf_sub[['cod_subcircu', 'subcircuito_nom', 'circuito_nom', 'distrito_nom', 'Area_km2', 'geometry']].merge(
        df_mensual_completo, on='cod_subcircu', how='right'
    )
    gdf_mensual['Tasa_Homicidios_km2'] = gdf_mensual['homicidios'] / gdf_mensual['Area_km2']
    
    # 7. Guardar bases de datos agregadas
    agregado_anual_path = os.path.join(config.CLEAN_DATA_DIR, "datos_agregados_anual.csv")
    agregado_mensual_path = os.path.join(config.CLEAN_DATA_DIR, "datos_agregados_mensual.csv")
    
    # Guardar como CSV (sin la columna geometry pesada)
    pd.DataFrame(gdf_anual.drop(columns=['geometry'])).to_csv(agregado_anual_path, index=False)
    pd.DataFrame(gdf_mensual.drop(columns=['geometry'])).to_csv(agregado_mensual_path, index=False)
    
    # Guardar df completo con las variables de ingeniería
    df.to_csv(os.path.join(config.CLEAN_DATA_DIR, "datos_limpios.csv"), index=False)
    
    # Guardar GeoJSON para mapas espaciales rápidos
    # Agrupamos por subcircuito histórico total para mapas ESDA
    homicidios_totales = df.groupby(config.COL_COD_SUBCIRCUITO).size().reset_index(name='homicidios_totales')
    homicidios_totales.columns = ['cod_subcircu', 'homicidios_totales']
    
    gdf_total = gdf_sub.merge(homicidios_totales, on='cod_subcircu', how='left').fillna({'homicidios_totales': 0})
    gdf_total['Tasa_Homicidios_km2'] = gdf_total['homicidios_totales'] / gdf_total['Area_km2']
    
    total_geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_homicidios.geojson")
    # Reproyectar a EPSG:4326 para Leaflet/Folium antes de guardar
    gdf_total.to_crs("EPSG:4326").to_file(total_geojson_path, driver='GeoJSON')
    
    logger.info(f"Bases agregadas anual/mensual guardadas en: {config.CLEAN_DATA_DIR}")
    logger.info(f"GeoJSON de subcircuitos guardado en: {total_geojson_path}")
    logger.info("ETAPA 3: Ingeniería de Variables y Agregación finalizada con éxito.")
    
    return df, gdf_total, gdf_anual, gdf_mensual
