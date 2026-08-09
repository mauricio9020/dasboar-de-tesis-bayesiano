import os
import sys
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def test_load():
    geojson_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_riesgo_completo.geojson")
    print("subcircuitos_riesgo_completo.geojson exists:", os.path.exists(geojson_path))
    
    esda_path = os.path.join(config.CLEAN_DATA_DIR, "subcircuitos_esda.geojson")
    print("subcircuitos_esda.geojson exists:", os.path.exists(esda_path))
    
    if os.path.exists(geojson_path):
        gdf = gpd.read_file(geojson_path)
    elif os.path.exists(esda_path):
        gdf = gpd.read_file(esda_path)
    else:
        gdf = None
        
    if gdf is not None:
        print("Columns:", gdf.columns.tolist())
        print("Shape:", gdf.shape)
        print("Null geometries:", gdf.geometry.isnull().sum())
        print("Head:")
        print(gdf.head(2))
    else:
        print("No gdf loaded")

if __name__ == "__main__":
    test_load()
