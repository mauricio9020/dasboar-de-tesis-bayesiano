import os
import sys
import logging
import time

# Agregar directorio actual al path para importar correctamente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar backend no interactivo para evitar fallos de Tkinter/hilos en Windows
import matplotlib
matplotlib.use('Agg')

from src import config
from src.stage1_loader import run_stage1
from src.stage2_cleaner import run_stage2
from src.stage3_engineering import run_stage3
from src.stage4_eda import run_stage4
from src.stage5_esda import run_stage5
from src.stage6_overdispersion import run_stage6
from src.stage7_classical import run_stage7
from src.stage8_bayesian import run_stage8
from src.stage9_diagnostics import run_stage9
from src.stage10_relative_risk import run_stage10
from src.stage11_space_time_dynamics import run_stage11
from src.stage13_exporter import run_stage13

# Configurar Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.PIPELINE_LOG_PATH, mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger("PipelineOrchestrator")

def main():
    start_time = time.time()
    logger.info("==============================================================")
    logger.info(" INICIANDO PIPELINE DE ANÁLISIS BAYESIANO ESPACIO-TEMPORAL ")
    logger.info("==============================================================")
    
    try:
        # Etapa 1: Cargar datos
        df_raw = run_stage1()
        
        # Etapa 2: Limpieza y Normalización
        df_clean = run_stage2(df_raw)
        
        # Etapa 3: Variable Engineering y Agregación
        df_eng, gdf_total, gdf_anual, gdf_mensual = run_stage3(df_clean)
        
        # Etapa 4: Análisis Exploratorio de Datos (EDA)
        stats_eda = run_stage4(df_eng)
        
        # Etapa 5: ESDA (Análisis Espacial Exploratorio y Autocorrelación)
        # Extraemos la matriz queen para pasársela a PyMC
        from src.stage5_esda import compare_weights
        w_selected, w_name, df_comp = compare_weights(gdf_total)
        
        gdf_esda, moran, df_comp, df_moran_global = run_stage5(gdf_total)
        
        # Etapa 6: Sobredispersión y Diagnóstico de Poisson
        stats_overdisp = run_stage6(gdf_anual)
        
        # Etapa 7: Ajuste de Modelos Clásicos (Poisson vs NegBin Frecuentista)
        poisson_res, nb_res, df_compare = run_stage7(gdf_anual)
        
        # Etapa 8: Ajuste y Comparación de los 4 Modelos Bayesianos mediante ADVI (PyMC) y Validación NUTS
        # Ajuste completo y real de los 4 modelos con ADVI (n=40000, tol=1e-4), 3 semillas, y validación NUTS
        bayesian_model, trace, df_comp_bayes, traces_dict = run_stage8(gdf_total, gdf_anual, w_selected)
        
        # Etapa 9: Diagnósticos Bayesianos y PPC
        metrics_info = run_stage9(trace, traces_dict=traces_dict)
        
        # Etapa 10: Estimación de Riesgos Relativos y Cartografía
        gdf_rr = run_stage10(trace, gdf_total, gdf_anual)
        
        # Etapa 11: Análisis Dinámico de Hotspots Espacio-Temporal
        df_evol = run_stage11(trace, gdf_anual)
        
        # Etapa 13: Exportación del Reporte Consolidado en Word
        # Obtenemos los subcircuitos más calientes ordenados para el Word
        df_rr_top = gdf_rr.sort_values(by='rr_estimado_mediana', ascending=False)
        word_path = run_stage13(stats_eda, df_comp, df_moran_global, df_compare, df_rr_top, df_evol)
        
        end_time = time.time()
        elapsed_min = (end_time - start_time) / 60
        
        logger.info("==============================================================")
        logger.info("    PIPELINE COMPLETADO CON ÉXITO EN {:.2f} MINUTOS".format(elapsed_min))
        logger.info(f"    Reporte Word generado en: {word_path}")
        logger.info("==============================================================")
        
    except Exception as e:
        logger.error("Error crítico durante la ejecución del pipeline:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
