from pathlib import Path
import os

# Directorio base del proyecto (un nivel arriba de src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Configurar w64devkit en PATH si está disponible en scratch/ para aceleración C++ en PyTensor
W64DEVKIT_BIN = BASE_DIR / "scratch" / "w64devkit" / "bin"
if W64DEVKIT_BIN.exists():
    os.environ["PATH"] = str(W64DEVKIT_BIN) + os.pathsep + os.environ.get("PATH", "")

# Semilla fija para reproducibilidad global (PyMC, permutaciones Moran, particiones)
RANDOM_SEED = 42

# Configuración de Rutas de Entrada
DATA_DIR = BASE_DIR / "data"
EXCEL_PATH = str(BASE_DIR / "Muertes_Violentas_Guayaquil_Nuevos_Solo_2024_2026.xlsx")

# Shapefile de Subcircuitos oficial en data/Subcircuitos/
SHAPEFILE_PATH = str(DATA_DIR / "Subcircuitos" / "Subcircuitos.shp")
if not os.path.exists(SHAPEFILE_PATH):
    # Fallback en caso de que esté en la raíz del proyecto
    alt_shapefile = BASE_DIR / "Subcircuitos.shp"
    if alt_shapefile.exists():
        SHAPEFILE_PATH = str(alt_shapefile)

# Configuración de Directorios de Salida
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = str(OUTPUT_DIR / "logs")
CLEAN_DATA_DIR = str(OUTPUT_DIR / "data")
FIGURES_DIR = str(OUTPUT_DIR / "figures")
MAPS_DIR = str(OUTPUT_DIR / "maps")
TABLES_DIR = str(OUTPUT_DIR / "tables")
MODELS_DIR = str(OUTPUT_DIR / "models")

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
PIPELINE_LOG_PATH = str(Path(LOGS_DIR) / "pipeline_execution.log")
