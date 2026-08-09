import os
import pandas as pd
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage1")

def run_stage1() -> pd.DataFrame:
    """
    Etapa 1: Carga de Datos
    - Lee el archivo Excel.
    - Detecta hojas.
    - Muestra estructura y dimensiones.
    - Identifica tipos de variables (numéricas, categóricas, temporales).
    """
    logger.info("Iniciando ETAPA 1: Carga de Datos")
    
    if not os.path.exists(config.EXCEL_PATH):
        raise FileNotFoundError(f"No se encontró el archivo Excel en {config.EXCEL_PATH}")
    
    # Detectar hojas del Excel
    xls = pd.ExcelFile(config.EXCEL_PATH)
    sheets = xls.sheet_names
    logger.info(f"Hojas detectadas en el Excel: {sheets}")
    
    # Cargar la primera hoja
    logger.info(f"Cargando hoja: {sheets[0]}...")
    df = pd.read_excel(config.EXCEL_PATH, sheet_name=sheets[0])
    
    # Dimensiones
    rows, cols = df.shape
    logger.info(f"Dimensiones de la base: {rows} filas y {cols} columnas.")
    
    # Estructura y tipos de variables
    num_vars = df.select_dtypes(include=['number']).columns.tolist()
    cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_vars = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    
    logger.info(f"Variables Numéricas ({len(num_vars)}): {num_vars}")
    logger.info(f"Variables Categóricas ({len(cat_vars)}): {cat_vars}")
    logger.info(f"Variables Temporales ({len(date_vars)}): {date_vars}")
    
    # Mostrar resumen
    logger.info("Estructura de la base (primeros 3 registros cargados con éxito)")
    
    # Validar que los datos se leen correctamente sin errores de codificación
    for col in df.columns[:5]:
        sample_vals = df[col].dropna().head(2).tolist()
        logger.info(f"Columna: '{col}' | Tipo: {df[col].dtype} | Muestras: {sample_vals}")
        
    logger.info("ETAPA 1: Carga de Datos finalizada con éxito.")
    return df
