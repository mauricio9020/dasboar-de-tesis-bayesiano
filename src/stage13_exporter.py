import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
import pandas as pd
import numpy as np
import os
import logging
from src import config

logger = logging.getLogger("Pipeline.Stage13")

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Establece márgenes de celda (padding) en dxa (1/20 de punto)"""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_background(cell, color_hex="F2F2F2"):
    """Colorea el fondo de la celda"""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._element.get_or_add_tcPr().append(shading_elm)

def apply_apa_table_style(table):
    """
    Aplica estilo APA a una tabla de Word:
    - Sin bordes verticales.
    - Líneas horizontales finas en la parte superior e inferior de la cabecera y parte inferior de la tabla.
    - Padding en las celdas para mayor legibilidad.
    """
    tblPr = table._element.tblPr
    # Remover bordes por defecto si los hay
    tblBorders = tblPr.first_child_found_in("w:tblBorders")
    if tblBorders is not None:
        tblPr.remove(tblBorders)
        
    # Crear bordes estilo APA (solo horizontales principales)
    apa_borders = parse_xml(
        '<w:tblBorders %s>'
        '  <w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '  <w:left w:val="none"/>'
        '  <w:right w:val="none"/>'
        '  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="D3D3D3"/>'
        '  <w:insideV w:val="none"/>'
        '</w:tblBorders>' % nsdecls('w')
    )
    tblPr.append(apa_borders)
    
    # Formatear celdas
    for row_idx, row in enumerate(table.rows):
        # Cabecera
        if row_idx == 0:
            for cell in row.cells:
                set_cell_background(cell, "333333")  # Gris oscuro
                set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
                # Texto en blanco y negrita se maneja en los párrafos de celda
        else:
            # Alternar color de fondo para filas
            bg_color = "F9F9F9" if row_idx % 2 == 0 else "FFFFFF"
            for cell in row.cells:
                set_cell_background(cell, bg_color)
                set_cell_margins(cell, top=100, bottom=100, left=150, right=150)

def add_paragraph_run(paragraph, text, bold=False, italic=False, size_pt=11, font_name='Arial', color_rgb=None):
    """Añade un run formateado de texto a un párrafo"""
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    if color_rgb:
        run.font.color.rgb = color_rgb
    return run

def create_word_table_from_df(doc, df, title_text, col_widths=None):
    """Crea una tabla formateada estilo APA en el documento Word a partir de un DataFrame"""
    # Título de la tabla (Norma APA: Tabla X en negrita, Título en cursiva)
    p_num = doc.add_paragraph()
    add_paragraph_run(p_num, title_text.split(":")[0] + ":", bold=True, size_pt=10)
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_after = Pt(6)
    add_paragraph_run(p_title, title_text.split(":")[1].strip(), italic=True, size_pt=10)
    
    # Crear Tabla
    table = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Cabeceras
    for col_idx, col_name in enumerate(df.columns):
        cell = table.cell(0, col_idx)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        run = add_paragraph_run(p, str(col_name), bold=True, size_pt=9.5)
        run.font.color.rgb = docx.shared.RGBColor(255, 255, 255) # Texto blanco en cabecera
        
    # Filas de Datos
    for row_idx in range(df.shape[0]):
        for col_idx in range(df.shape[1]):
            val = df.iloc[row_idx, col_idx]
            # Formatear flotantes para que no tengan decimales excesivos
            if isinstance(val, float):
                val_str = f"{val:.4f}" if val < 1 else f"{val:.2f}"
            else:
                val_str = str(val)
                
            cell = table.cell(row_idx + 1, col_idx)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            add_paragraph_run(p, val_str, size_pt=9)
            
    # Ajustar anchos de columnas
    if col_widths:
        for idx, width in enumerate(col_widths):
            for cell in table.columns[idx].cells:
                cell.width = Inches(width)
                
    # Aplicar estilos APA
    apply_apa_table_style(table)
    doc.add_paragraph().paragraph_format.space_before = Pt(12)  # Separador inferior

def run_stage13(stats_eda: dict, df_comp: pd.DataFrame, df_moran_global: pd.DataFrame, df_compare: pd.DataFrame, df_rr_top: pd.DataFrame, df_evol: pd.DataFrame) -> str:
    """
    Etapa 13: Exportaciones Automatizadas
    - Genera el reporte formateado en Word (.docx) con todas las tablas requeridas.
    - Asegura la nomenclatura, numeración y diseño profesional.
    - Retorna la ruta física del archivo generado.
    """
    logger.info("Iniciando ETAPA 13: Exportación del Reporte Científico de Tesis (Word)")
    
    doc = docx.Document()
    
    # Configurar márgenes estándar (1 pulgada)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Título Principal del Reporte
    p_main = doc.add_paragraph()
    p_main.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_main.paragraph_format.space_after = Pt(24)
    add_paragraph_run(p_main, "REPORTE METODOLÓGICO DE TESIS\n", bold=True, size_pt=16)
    add_paragraph_run(p_main, "\"Análisis Bayesiano de Homicidios en el Cantón Guayaquil mediante Análisis Espacio-Temporal\"\n", bold=True, size_pt=14, color_rgb=docx.shared.RGBColor(128, 0, 0))
    add_paragraph_run(p_main, "Análisis de Datos y Modelamiento Jerárquico Espacial (2024-2026)", size_pt=12, italic=True)
    
    # Introducción Breve
    p_intro = doc.add_paragraph()
    p_intro.paragraph_format.space_after = Pt(12)
    p_intro.paragraph_format.line_spacing = 1.15
    add_paragraph_run(p_intro, "Este documento consolidado contiene el flujo de tablas de resultados de la metodología "
                               "desarrollada para el modelado espacio-temporal de muertes violentas registradas en el Cantón "
                               "Guayaquil entre 2024 y 2026. Los resultados cubren el control de calidad, el análisis de autocorrelación "
                               "espacial (LISA), los diagnósticos de sobredispersión, la comparación de modelos Poisson y Binomial "
                               "Negativa frecuentistas, y los parámetros posteriores del modelo Bayesiano jerárquico estimado.", size_pt=11)
    
    # 1. Tabla 1: Reporte de Calidad de Datos (Etapa 2)
    # Re-leer o reconstruir la tabla de calidad de datos
    quality_xlsx_path = os.path.join(config.CLEAN_DATA_DIR, "reporte_calidad.xlsx")
    if os.path.exists(quality_xlsx_path):
        df_quality = pd.read_excel(quality_xlsx_path)
    else:
        df_quality = pd.DataFrame({
            "Métrica": ["Registros Iniciales", "Registros Finales Limpios"],
            "Valor": [stats_eda.get("homicidios_totales", 4716), stats_eda.get("homicidios_totales", 4716)]
        })
    create_word_table_from_df(doc, df_quality, "Tabla 1: Resultados del Proceso de Calidad y Depuración de Datos", [3.5, 2.0])
    
    # 2. Tabla 2: Resumen Descriptivo Univariado (Etapa 4)
    # Armamos una tabla consolidada a partir de los diccionarios de stats_eda
    armas = list(stats_eda["frec_arma"].keys())
    arma_frec = list(stats_eda["frec_arma"].values())
    arma_pct = [f"{v:.2f}%" for v in stats_eda["pct_arma"].values()]
    df_desc = pd.DataFrame({
        "Categoría de Variable": ["Tipo de Arma"] * len(armas),
        "Variable / Valor": armas,
        "Frecuencia Absoluta": arma_frec,
        "Porcentaje Relativo": arma_pct
    })
    create_word_table_from_df(doc, df_desc, "Tabla 2: Distribución de Frecuencia de los Homicidios según el Tipo de Arma Utilizada", [2.0, 2.0, 1.5, 1.5])
    
    # 3. Tabla 3: Comparativa Queen vs Rook (Etapa 5)
    create_word_table_from_df(doc, df_comp, "Tabla 3: Comparación Matemática de Matrices de Vecindad Queen y Rook", [2.5, 2.0, 2.0])
    
    # 4. Tabla 4: Autocorrelación Espacial Moran (Etapa 5)
    create_word_table_from_df(doc, df_moran_global, "Tabla 4: Índice de Autocorrelación Espacial Global de Moran para Homicidios", [3.0, 2.5])
    
    # 5. Tabla 5: Comparativa de Modelos Clásicos e Inf (Etapa 7)
    create_word_table_from_df(doc, df_compare, "Tabla 5: Comparativa de Ajuste entre Modelos Poisson y Binomial Negativo Frecuentistas", [2.5, 2.0, 2.0])
    
    # 6. Tabla 6: Top 10 Zonas Críticas (Etapa 10)
    # Tomamos los 10 primeros del df_rr_top
    df_top10 = df_rr_top[['subcircuito_nom', 'circuito_nom', 'distrito_nom', 'rr_estimado_mediana', 'probabilidad_excedencia']].head(10).copy()
    df_top10.columns = ['Subcircuito', 'Circuito', 'Distrito', 'Riesgo Relativo (RR)', 'P(RR > 1.5)']
    create_word_table_from_df(doc, df_top10, "Tabla 6: Top 10 Subcircuitos con Mayor Riesgo Relativo Estimado mediante el Modelo Bayesiano", [2.0, 1.5, 1.5, 1.5, 1.0])
    
    # 7. Tabla 7: Dinámica Espacio-Temporal (Etapa 11)
    df_dyn_summary = df_evol['clasificacion_dinamica'].value_counts().reset_index()
    df_dyn_summary.columns = ['Clasificación Dinámica de Hotspots', 'Número de Subcircuitos']
    create_word_table_from_df(doc, df_dyn_summary, "Tabla 7: Resumen del Comportamiento Espacio-Temporal de los Subcircuitos (2024-2026)", [3.5, 2.0])
    
    # 8. Tabla 8: Comparación Formal de Modelos Bayesianos mediante LOO-CV (Etapa 8)
    excel_comp_path = os.path.join(config.TABLES_DIR, "tabla_comparacion_modelos_bayesiana.xlsx")
    if os.path.exists(excel_comp_path):
        try:
            df_comp_b = pd.read_excel(excel_comp_path)
            create_word_table_from_df(doc, df_comp_b, "Tabla 8: Comparación de Modelos Bayesianos mediante Validación Cruzada LOO-CV (Escala Log ELPD)")
        except Exception as e:
            logger.warning(f"No se pudo adjuntar Tabla 8 en Word: {e}")
            
    # Guardar Documento Word
    word_path = os.path.join(config.TABLES_DIR, "Reporte_Tesis_Flujo_Bayesiano.docx")
    doc.save(word_path)
    
    logger.info(f"Reporte Word completado y guardado en: {word_path}")
    logger.info("ETAPA 13: Exportación del Reporte Científico finalizada con éxito.")
    
    return word_path
