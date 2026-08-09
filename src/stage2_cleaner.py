import pandas as pd
import numpy as np
import logging
import os
import unicodedata
from src import config

logger = logging.getLogger("Pipeline.Stage2")

def remove_accents(text: str) -> str:
    """Remueve tildes y caracteres especiales, preservando la letra Ñ"""
    if not isinstance(text, str):
        return text
    # Estandarizar tildes comunes
    translation = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ü': 'u', 'Ü': 'U'
    }
    for acc, clean in translation.items():
        text = text.replace(acc, clean)
    return text

def run_stage2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Etapa 2: Limpieza de Datos
    - Eliminar duplicados.
    - Normalizar tildes, mayúsculas y espacios.
    - Estandarizar coordenadas y filtrar por Bounding Box.
    - Convertir formatos de fecha y hora.
    - Crear campos temporales derivados (Año, Mes, Trimestre, Semana, Día de la Semana, Hora).
    - Generar identificadores únicos.
    - Guardar base limpia y reporte de calidad.
    """
    logger.info("Iniciando ETAPA 2: Limpieza y Normalización")
    initial_rows = len(df)
    
    # 1. Eliminar duplicados
    df = df.drop_duplicates()
    duplicate_count = initial_rows - len(df)
    logger.info(f"Duplicados eliminados: {duplicate_count} registros.")
    
    # 2. Normalización de tildes, mayúsculas y espacios en columnas categóricas
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Normalizando texto en columnas: {text_cols}")
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].apply(remove_accents)
        # Reemplazar valores nulos simulados como 'NAN', 'NAT', 'NONE' por nulos reales
        df[col] = df[col].replace({'NAN': np.nan, 'NAT': np.nan, 'NONE': np.nan, 'SIN_DATO': np.nan, 'SIN DATO': np.nan})
        
    # 3. Limpieza y conversión de Coordenadas
    logger.info("Estandarizando coordenadas...")
    
    # Limpiar y convertir latitud (Coord. Y)
    df[config.COL_COORD_Y] = df[config.COL_COORD_Y].astype(str).str.replace(',', '.')
    df[config.COL_COORD_Y] = pd.to_numeric(df[config.COL_COORD_Y], errors='coerce')
    
    # Limpiar y convertir longitud (Coord. X)
    df[config.COL_COORD_X] = df[config.COL_COORD_X].astype(str).str.replace(',', '.')
    df[config.COL_COORD_X] = pd.to_numeric(df[config.COL_COORD_X], errors='coerce')
    
    # Identificar registros sin coordenadas válidas
    missing_coords = df[df[config.COL_COORD_X].isnull() | df[config.COL_COORD_Y].isnull()]
    missing_coords_count = len(missing_coords)
    
    # Filtrar coordenadas válidas usando el Bounding Box de Guayaquil
    in_box = (
        (df[config.COL_COORD_Y] >= config.LAT_MIN) & (df[config.COL_COORD_Y] <= config.LAT_MAX) &
        (df[config.COL_COORD_X] >= config.LON_MIN) & (df[config.COL_COORD_X] <= config.LON_MAX)
    )
    out_of_box = df[~in_box & df[config.COL_COORD_X].notnull() & df[config.COL_COORD_Y].notnull()]
    out_of_box_count = len(out_of_box)
    
    # Mantener solo registros dentro del Bounding Box
    df = df[in_box].copy()
    cleaned_rows = len(df)
    removed_coords_count = initial_rows - duplicate_count - cleaned_rows
    
    logger.info(f"Coordenadas nulas: {missing_coords_count} registros.")
    logger.info(f"Coordenadas fuera de rango de Guayaquil: {out_of_box_count} registros.")
    logger.info(f"Registros eliminados por problemas geográficos: {removed_coords_count} registros.")
    
    # Renombrar columnas geográficas estándar para modelado
    df['latitud'] = df[config.COL_COORD_Y]
    df['longitud'] = df[config.COL_COORD_X]
    
    # 4. Formateo y conversión de Fecha y Hora
    logger.info("Procesando fechas y horas...")
    df['fecha_infraccion_dt'] = pd.to_datetime(df[config.COL_FECHA], errors='coerce')
    
    # Rellenar fechas nulas si existieran (no debería haber, pero es una salvaguarda)
    df = df.dropna(subset=['fecha_infraccion_dt'])
    
    # Crear variables temporales detalladas
    df['Anio'] = df['fecha_infraccion_dt'].dt.year
    df['Mes_Num'] = df['fecha_infraccion_dt'].dt.month
    df['Mes'] = df['fecha_infraccion_dt'].dt.month_name(locale='es_ES').str.upper().apply(remove_accents)
    df['Trimestre'] = df['fecha_infraccion_dt'].dt.quarter
    df['Semana'] = df['fecha_infraccion_dt'].dt.isocalendar().week
    
    # Días de la semana en español
    dia_semana_map = {
        0: 'LUNES', 1: 'MARTES', 2: 'MIERCOLES', 3: 'JUEVES',
        4: 'VIERNES', 5: 'SABADO', 6: 'DOMINGO'
    }
    df['DiaSemana'] = df['fecha_infraccion_dt'].dt.dayofweek.map(dia_semana_map)
    
    # Extraer Hora
    # Si la columna Hora original es numérica, usarla, si no parsear Hora Infracción
    if 'Hora' in df.columns:
        df['Hora'] = pd.to_numeric(df['Hora'], errors='coerce').fillna(0).astype(int)
    else:
        # Extraer del string de Hora Infracción (formato HH:MM:SS)
        df['Hora'] = df[config.COL_HORA_TXT].astype(str).str.split(':').str[0]
        df['Hora'] = pd.to_numeric(df['Hora'], errors='coerce').fillna(0).astype(int)
    
    # 5. Generar Identificadores Únicos
    df = df.sort_values(by='fecha_infraccion_dt').reset_index(drop=True)
    df['ID_Homicidio'] = [f"MV-GYE-{i+1:05d}" for i in range(len(df))]
    
    # 6. Reporte de Calidad de Datos
    quality_summary = {
        "Métrica": [
            "Registros Iniciales",
            "Duplicados Eliminados",
            "Coordenadas Nulas Eliminadas",
            "Coordenadas Fuera de Bounding Box Eliminadas",
            "Registros Limpios Finales",
            "Valores Faltantes en Edad",
            "Valores Faltantes en Arma",
            "Valores Faltantes en Presunto Móvil",
            "Valores Faltantes en Antecedentes"
        ],
        "Valor": [
            initial_rows,
            duplicate_count,
            missing_coords_count,
            out_of_box_count,
            cleaned_rows,
            df[config.COL_EDAD].isnull().sum(),
            df[config.COL_ARMA].isnull().sum(),
            df[config.COL_PRESUN_MOTIVACION].isnull().sum(),
            df['Antecedentes'].isnull().sum() if 'Antecedentes' in df.columns else 0
        ]
    }
    df_quality = pd.DataFrame(quality_summary)
    
    # Guardar Reportes
    clean_csv_path = os.path.join(config.CLEAN_DATA_DIR, "datos_limpios.csv")
    quality_xlsx_path = os.path.join(config.CLEAN_DATA_DIR, "reporte_calidad.xlsx")
    
    df.to_csv(clean_csv_path, index=False, encoding='utf-8')
    df_quality.to_excel(quality_xlsx_path, index=False)
    
    logger.info(f"Base limpia guardada en: {clean_csv_path}")
    logger.info(f"Reporte de calidad guardado en: {quality_xlsx_path}")
    logger.info("ETAPA 2: Limpieza y Normalización finalizada con éxito.")
    
    return df
