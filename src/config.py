import os

# Obtener el directorio base del proyecto (un nivel arriba de src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuración de Rutas de Entrada
DATA_DIR = BASE_DIR
EXCEL_PATH = os.path.join(DATA_DIR, "Muertes_Violentas_Guayaquil_Nuevos_Solo_2024_2026.xlsx")

# Buscar el shapefile en el directorio del proyecto, o usar la ruta local original como fallback
SHAPEFILE_PATH = os.path.join(DATA_DIR, "Subcircuitos.shp")
if not os.path.exists(SHAPEFILE_PATH):
    SHAPEFILE_PATH = r"c:\Users\ASUS\Downloads\holamundo\Bases de datos separadas\Subcircuitos shp\Subcircuitos.shp"

# Configuración de Directorios de Salida
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CLEAN_DATA_DIR = os.path.join(OUTPUT_DIR, "data")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Crear directorios automáticamente si no existen
for directory in [OUTPUT_DIR, LOGS_DIR, CLEAN_DATA_DIR, FIGURES_DIR, MAPS_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Bounding box para Guayaquil (Urbano y Rural)
LAT_MIN, LAT_MAX = -3.0, -1.9
LON_MIN, LON_MAX = -80.3, -79.6

# Nombres de las columnas originales
COL_TIPO_MUERTE = 'Tipo Muert.'
COL_DISTRITO = 'Distrito'
COL_CIRCUITO = 'Circuito'
COL_COD_SUBCIRCUITO = 'Cod. Subcircu'
COL_SUBCIRCUITO = 'Subcircuito'
COL_COORD_Y = 'Coord. Y'
COL_COORD_X = 'Coord. X'
COL_FECHA = 'Fecha Infracción'
COL_HORA_TXT = 'Hora Infracción'
COL_ARMA = 'Arma'
COL_TIPO_ARMA = 'Tipo Arma'
COL_PRESUN_MOTIVACION = 'Presun. Motiva.'
COL_PRESUN_MOTIVACION_OBS = 'Presun. Motiva. Obser'
COL_EDAD = 'Edad'
COL_SEXO = 'Sexo'
COL_RANGO_EDAD = 'Rango Edad'

# Archivo de logs del Pipeline
PIPELINE_LOG_PATH = os.path.join(LOGS_DIR, "pipeline_execution.log")
