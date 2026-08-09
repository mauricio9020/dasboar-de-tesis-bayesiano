import os
import sys
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from shapely.geometry import Point
import matplotlib.patches as patches

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
    
    # 2. Cargar y disolver límites parroquiales
    print("\nProcesando límites parroquiales desde el shapefile...")
    gdf_parr = None
    if os.path.exists(config.SHAPEFILE_PATH):
        try:
            gdf_shp = gpd.read_file(config.SHAPEFILE_PATH)
            # Filtrar por Guayaquil
            gdf_gye_shp = gdf_shp[gdf_shp['DPA_DESCAN'].str.upper() == 'GUAYAQUIL'].copy()
            if gdf_gye_shp.crs is None:
                gdf_gye_shp = gdf_gye_shp.set_crs("EPSG:32717")
            # Disolver por parroquia para agrupar
            gdf_parr = gdf_gye_shp.dissolve(by='DPA_PARROQ', as_index=False)
            print(f"[OK] Se cargaron {len(gdf_parr)} parroquias de Guayaquil.")
        except Exception as e:
            print(f"Error al procesar el shapefile: {e}")
            gdf_parr = None
    else:
        print(f"Advertencia: No se encontró el shapefile en {config.SHAPEFILE_PATH}. No se dibujarán los límites parroquiales.")

    # -------------------------------------------------------------
    # PARTE 1: GENERACIÓN DEL MAPA DE CALOR INTERACTIVO (FOLIUM HTML)
    # -------------------------------------------------------------
    print("\nGenerando mapa de calor interactivo (KDE) en HTML...")
    
    # Crear el mapa base sin tiles por defecto
    m = folium.Map(
        location=[lat_mean, lon_mean],
        zoom_start=12,
        tiles=None
    )
    
    # Capas base muy tenues
    folium.TileLayer('cartodbpositron', name='CartoDB Positron (Gris Claro)', control=True).add_to(m)
    folium.TileLayer('cartodb_dark_matter', name='CartoDB Dark Matter (Oscuro)', control=True).add_to(m)
    folium.TileLayer('openstreetmap', name='OpenStreetMap (Color)', control=True).add_to(m)
    
    # Preparar datos para el HeatMap de Folium: lista de [latitud, longitud]
    heat_data = df[['latitud', 'longitud']].values.tolist()
    
    # Gradiente continuo de Azul -> Verde -> Amarillo -> Naranja -> Rojo
    custom_gradient = {
        0.0: '#2b83ba',  # Azul (baja densidad)
        0.25: '#abdda4', # Verde
        0.5: '#ffffbf',  # Amarillo (densidad media)
        0.75: '#fdae61', # Naranja
        1.0: '#d7191c'   # Rojo intenso (alta concentración)
    }
    
    # Mapa de calor interactivo con radio ajustado a 28 y blur a 20
    # Esto reduce la sobre-suavización y desconecta Posorja visualmente
    heatmap_layer = HeatMap(
        heat_data,
        name="Estimación de Densidad Kernel (KDE)",
        min_opacity=0.25,
        max_zoom=18,
        radius=28,       # Radio optimizado
        blur=20,         # Suavizado optimizado
        gradient=custom_gradient,
        show=True
    )
    heatmap_layer.add_to(m)
    
    # Añadir límites parroquiales al mapa interactivo si están disponibles
    if gdf_parr is not None:
        gdf_parr_wgs84 = gdf_parr.to_crs(epsg=4326)
        folium.GeoJson(
            gdf_parr_wgs84,
            name="Límites Parroquiales",
            style_function=lambda x: {
                'fillColor': 'none',
                'color': '#555555',
                'weight': 1.2,
                'dashArray': '4, 4',
                'fillOpacity': 0.0
            },
            tooltip=folium.GeoJsonTooltip(fields=['NOMPARROQ'], aliases=['Parroquia:']),
            show=True
        ).add_to(m)
    
    # Capa de puntos individuales opcional
    points_layer = folium.FeatureGroup(name="Homicidios (Puntos de Eventos)", show=False)
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=1.5,
            color='black',
            weight=0.5,
            fill=True,
            fill_color='black',
            fill_opacity=0.3
        ).add_to(points_layer)
    points_layer.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Guardar mapa interactivo
    interactive_kde_path = os.path.join(config.MAPS_DIR, "mapa_homicidios_kde_interactivo.html")
    m.save(interactive_kde_path)
    print(f"[OK] Mapa interactivo de calor guardado en: {interactive_kde_path}")
    
    # -------------------------------------------------------------
    # PARTE 2: GENERACIÓN DE MAPAS DE CALOR ESTÁTICOS (PNG 300 DPI)
    # -------------------------------------------------------------
    print("\nGenerando mapas KDE estáticos con límites parroquiales, leyenda, norte y escala...")
    
    # Convertir homicidios a GeoDataFrame y proyectar a Web Mercator (EPSG:3857)
    geometry = [Point(xy) for xy in zip(df['longitud'], df['latitud'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf_mercator = gdf.to_crs(epsg=3857)
    
    # Límites del encuadre basados en los homicidios (con 5% de margen)
    # Así evitamos que el mapa se aleje demasiado por incluir parroquias completas alejadas
    xmin, ymin, xmax, ymax = gdf_mercator.total_bounds
    x_pad = (xmax - xmin) * 0.05
    y_pad = (ymax - ymin) * 0.05
    xlim = (xmin - x_pad, xmax + x_pad)
    ylim = (ymin - y_pad, ymax + y_pad)
    
    # Proyectar límites parroquiales a Web Mercator si están cargados
    gdf_parr_mercator = None
    if gdf_parr is not None:
        gdf_parr_mercator = gdf_parr.to_crs(epsg=3857)
    
    # Intentar cargar CartoDB Voyager si está disponible en contextily, si no usar Positron
    try:
        voyager_provider = cx.providers.CartoDB.Voyager
    except AttributeError:
        voyager_provider = cx.providers.CartoDB.Positron
        
    configurations = [
        {
            "name": "mapa_homicidios_kde_spectral_cartodb.png",
            "basemap": cx.providers.CartoDB.Positron,
            "cmap": "Spectral_r",
            "title": "Estimación de Densidad Kernel (KDE) - Spectral"
        },
        {
            "name": "mapa_homicidios_kde_viridis_cartodb.png",
            "basemap": cx.providers.CartoDB.Positron,
            "cmap": "viridis",
            "title": "Estimación de Densidad Kernel (KDE) - Viridis"
        },
        {
            "name": "mapa_homicidios_kde_voyager.png",
            "basemap": voyager_provider,
            "cmap": "Spectral_r",
            "title": "Estimación de Densidad Kernel (KDE) - CartoDB Voyager"
        }
    ]
    
    for config_item in configurations:
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 1. Graficar el KDE con parámetros corregidos:
        # - bw_adjust=1.35: reduce ligeramente el ancho de banda para independizar Posorja
        # - thresh=0.02: corta la densidad muy baja en los extremos para que no se unan los halos
        sns.kdeplot(
            x=gdf_mercator.geometry.x,
            y=gdf_mercator.geometry.y,
            ax=ax,
            cmap=config_item["cmap"],
            fill=True,
            alpha=0.6,
            thresh=0.02,         # Ajustado para cortar la densidad baja residual
            levels=20,
            gridsize=300,
            bw_adjust=1.35,      # Ajustado para independizar Posorja
            cbar=True,
            cbar_kws={'shrink': 0.5, 'pad': 0.02},
            zorder=3
        )
        
        # 2. Graficar los límites parroquiales como líneas grises muy tenues por encima
        if gdf_parr_mercator is not None:
            gdf_parr_mercator.plot(
                ax=ax,
                facecolor='none',
                edgecolor='#555555',
                linewidth=0.6,
                linestyle='--',
                alpha=0.55,
                zorder=3.5       # Por encima de la densidad pero por debajo de la escala/norte
            )
            
        # 3. Agregar mapa base muy tenue (alpha=0.7)
        cx.add_basemap(ax, source=config_item["basemap"], alpha=0.7, zorder=1)
        
        # 4. Personalizar la Leyenda (Colorbar)
        cbar_ax = fig.axes[-1]
        cbar_ax.set_yticks([]) # Quitar valores numéricos absolutos
        cbar_ax.set_ylabel('Densidad de Homicidios (Relativa)', fontsize=12, fontweight='bold', labelpad=15)
        cbar_ax.text(1.2, 0.02, 'Baja', ha='left', va='center', fontsize=10, transform=cbar_ax.transAxes, fontweight='bold')
        cbar_ax.text(1.2, 0.98, 'Alta', ha='left', va='center', fontsize=10, transform=cbar_ax.transAxes, fontweight='bold')
        
        # Ajustar encuadre
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # 5. Dibujar la Escala de Barra (5 km) con fondo blanco protector
        scale_len = 5000  # metros (5 km)
        scale_x_start = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        scale_y = ylim[0] + (ylim[1] - ylim[0]) * 0.04
        
        tick_height = (ylim[1] - ylim[0]) * 0.01
        text_offset = tick_height * 1.2
        
        # Fondo blanco para la escala
        box_x = scale_x_start - 500
        box_y = scale_y - tick_height
        box_w = scale_len + 1000
        box_h = tick_height * 3.5
        
        scale_box = patches.Rectangle(
            (box_x, box_y), box_w, box_h,
            linewidth=1, edgecolor='#cccccc', facecolor='white', alpha=0.85, zorder=4
        )
        ax.add_patch(scale_box)
        
        # Línea y marcas de la escala
        ax.plot([scale_x_start, scale_x_start + scale_len], [scale_y, scale_y], color='black', linewidth=3, zorder=5)
        ax.plot([scale_x_start, scale_x_start], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
        ax.plot([scale_x_start + scale_len/2, scale_x_start + scale_len/2], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
        ax.plot([scale_x_start + scale_len, scale_x_start + scale_len], [scale_y - tick_height/2, scale_y + tick_height/2], color='black', linewidth=1.5, zorder=5)
        
        # Etiquetas de escala
        ax.text(scale_x_start, scale_y + text_offset, '0', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
        ax.text(scale_x_start + scale_len/2, scale_y + text_offset, '2.5 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
        ax.text(scale_x_start + scale_len, scale_y + text_offset, '5 km', ha='center', fontsize=9, fontweight='bold', color='black', zorder=6)
        
        # 6. Dibujar la Flecha del Norte con fondo blanco protector
        rect_n = patches.Rectangle(
            (0.88, 0.85), 0.08, 0.11,
            transform=ax.transAxes,
            linewidth=1, edgecolor='#cccccc', facecolor='white', alpha=0.85, zorder=4
        )
        ax.add_patch(rect_n)
        
        ax.annotate(
            'N', xy=(0.92, 0.93), xytext=(0.92, 0.87),
            transform=ax.transAxes,
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=3, headwidth=10, shrink=0.1),
            ha='center', va='center', fontsize=14, fontweight='bold', color='black',
            zorder=6
        )
        
        # Limpieza estética del mapa (apagar ejes)
        ax.set_axis_off()
        
        # Guardar imagen a 300 DPI
        out_img_path = os.path.join(config.FIGURES_DIR, config_item["name"])
        plt.savefig(out_img_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"[OK] Mapa KDE estático guardado en: {out_img_path}")
        
    print("\n" + "=" * 60)
    print("PROCESO TERMINADO CON ÉXITO")
    print("=" * 60)

if __name__ == "__main__":
    main()
