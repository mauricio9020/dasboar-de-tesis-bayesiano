import os
import sys
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import Point

# Agregar el directorio actual al path para importar configuración
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config

def main():
    print("=" * 60)
    # 1. Cargar datos limpios
    clean_csv_path = os.path.join(config.CLEAN_DATA_DIR, "datos_limpios.csv")
    if not os.path.exists(clean_csv_path):
        print(f"Error: No se encontró el archivo de datos limpios en: {clean_csv_path}")
        print("Por favor ejecuta primero 'run_pipeline.py' para procesar los datos.")
        return
        
    print(f"Cargando datos desde: {clean_csv_path}")
    df = pd.read_csv(clean_csv_path)
    
    # Filtrar coordenadas nulas por seguridad
    df = df.dropna(subset=['latitud', 'longitud'])
    print(f"Total de registros a mapear: {len(df)}")
    
    # Crear carpetas de salida si no existen
    os.makedirs(config.MAPS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Coordenadas promedio para centrar el mapa interactivo
    lat_mean = df['latitud'].mean()
    lon_mean = df['longitud'].mean()
    print(f"Centro geográfico estimado: Latitud={lat_mean:.5f}, Longitud={lon_mean:.5f}")
    
    # -------------------------------------------------------------
    # PARTE 1: GENERACIÓN DEL MAPA INTERACTIVO (FOLIUM HTML)
    # -------------------------------------------------------------
    print("\nGenerando mapa interactivo en HTML...")
    
    # Crear el mapa base con CartoDB Positron por defecto (muy limpio y elegante)
    m = folium.Map(
        location=[lat_mean, lon_mean],
        zoom_start=12,
        tiles=None # No cargar tiles por defecto para controlarlos con LayerControl
    )
    
    # Añadir capas base (Basemaps)
    folium.TileLayer('openstreetmap', name='OpenStreetMap', control=True).add_to(m)
    folium.TileLayer('cartodbpositron', name='CartoDB Positron (Luz)', control=True).add_to(m)
    folium.TileLayer('cartodb_dark_matter', name='CartoDB Dark Matter (Oscuro)', control=True).add_to(m)
    
    # Crear grupos de capas para los colores de los puntos
    group_red = folium.FeatureGroup(name="Homicidios (Puntos Rojos)", show=True)
    group_blue = folium.FeatureGroup(name="Homicidios (Puntos Azules)", show=False)
    
    # Agregar los puntos a cada capa
    for idx, row in df.iterrows():
        lat = row['latitud']
        lon = row['longitud']
        fecha = row.get('Fecha Infracción', 'N/A')
        arma = row.get('Arma', 'N/A')
        distrito = row.get('Distrito', 'N/A')
        
        # Info básica para el popup interactivo
        popup_text = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; line-height: 1.5;">
            <b>ID:</b> {row.get('ID_Homicidio', idx)}<br>
            <b>Fecha:</b> {fecha}<br>
            <b>Arma:</b> {arma}<br>
            <b>Distrito:</b> {distrito}<br>
            <b>Coordenadas:</b> {lat:.5f}, {lon:.5f}
        </div>
        """
        
        # Puntos Rojos
        folium.CircleMarker(
            location=[lat, lon],
            radius=2.5,
            popup=folium.Popup(popup_text, max_width=250),
            color='#d32f2f',       # Borde rojo oscuro
            fill=True,
            fill_color='#ff1744',  # Relleno rojo brillante
            fill_opacity=0.8,
            weight=0.8
        ).add_to(group_red)
        
        # Puntos Azules
        folium.CircleMarker(
            location=[lat, lon],
            radius=2.5,
            popup=folium.Popup(popup_text, max_width=250),
            color='#0d47a1',       # Borde azul oscuro
            fill=True,
            fill_color='#2979ff',  # Relleno azul brillante
            fill_opacity=0.8,
            weight=0.8
        ).add_to(group_blue)
        
    # Añadir los grupos de puntos al mapa
    group_red.add_to(m)
    group_blue.add_to(m)
    
    # Añadir el control de capas (permite alternar fondos y color de puntos)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Guardar mapa interactivo
    interactive_map_path = os.path.join(config.MAPS_DIR, "mapa_homicidios_puntos_interactivo.html")
    m.save(interactive_map_path)
    print(f"[OK] Mapa interactivo guardado en: {interactive_map_path}")
    
    # -------------------------------------------------------------
    # PARTE 2: GENERACIÓN DE MAPAS ESTÁTICOS (PNG DE ALTA RESOLUCIÓN)
    # -------------------------------------------------------------
    print("\nGenerando mapas estáticos en PNG de alta resolución (300 DPI) para la tesis...")
    
    # Convertir a GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['longitud'], df['latitud'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Proyectar a Web Mercator (EPSG:3857) que es el sistema requerido por Contextily
    gdf_mercator = gdf.to_crs(epsg=3857)
    
    # Obtener límites de Guayaquil con un 5% de margen para encuadrar bien el mapa
    xmin, ymin, xmax, ymax = gdf_mercator.total_bounds
    x_pad = (xmax - xmin) * 0.05
    y_pad = (ymax - ymin) * 0.05
    xlim = (xmin - x_pad, xmax + x_pad)
    ylim = (ymin - y_pad, ymax + y_pad)
    
    # Configuración de combinaciones de mapas estáticos
    combinations = [
        {
            "name": "mapa_homicidios_puntos_rojos_osm.png",
            "color": "#e31a1c",
            "basemap": cx.providers.OpenStreetMap.Mapnik,
            "title": "Mapa de Puntos de Homicidios - Guayaquil (OpenStreetMap / Puntos Rojos)"
        },
        {
            "name": "mapa_homicidios_puntos_azules_osm.png",
            "color": "#1f78b4",
            "basemap": cx.providers.OpenStreetMap.Mapnik,
            "title": "Mapa de Puntos de Homicidios - Guayaquil (OpenStreetMap / Puntos Azules)"
        },
        {
            "name": "mapa_homicidios_puntos_rojos_cartodb.png",
            "color": "#e31a1c",
            "basemap": cx.providers.CartoDB.Positron,
            "title": "Mapa de Puntos de Homicidios - Guayaquil (CartoDB Positron / Puntos Rojos)"
        },
        {
            "name": "mapa_homicidios_puntos_azules_cartodb.png",
            "color": "#1f78b4",
            "basemap": cx.providers.CartoDB.Positron,
            "title": "Mapa de Puntos de Homicidios - Guayaquil (CartoDB Positron / Puntos Azules)"
        }
    ]
    
    for combo in combinations:
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Graficar los puntos
        gdf_mercator.plot(
            ax=ax,
            color=combo["color"],
            markersize=3,        # Tamaño óptimo de puntos para evitar solapamientos masivos
            alpha=0.6,           # Transparencia para ver densidad donde hay más solapamientos
            edgecolor='none'
        )
        
        # Agregar mapa base
        cx.add_basemap(ax, source=combo["basemap"])
        
        # Ajustar los límites del encuadre
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Apagar ejes para una cartografía limpia en la tesis
        ax.set_axis_off()
        
        # Guardar imagen a 300 DPI
        out_img_path = os.path.join(config.FIGURES_DIR, combo["name"])
        plt.savefig(out_img_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"[OK] Mapa estático guardado en: {out_img_path}")
        
    print("\n" + "=" * 60)
    print("PROCESO TERMINADO CON ÉXITO")
    print(f"Los mapas interactivos se guardaron en: {config.MAPS_DIR}")
    print(f"Las imágenes estáticas se guardaron en: {config.FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
