import streamlit as st
import os
import re
import json
import torch
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt
import contextily as cx
from typing import Dict, Tuple, Optional, Union, Any, List, Literal
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union, linemerge
from pyproj import CRS
from scipy.ndimage import generic_filter
from collections import deque
import shutil
import traceback
import zipfile
import time  # For runtime measurement

# LangChain specific imports
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field, ValidationError

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Advanced Geospatial Flood Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Environment Setup ---
# Define your working data directory where files will be unzipped or placed
# For Streamlit, assume 'data' directory is co-located with the app.py
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_DIR = "flood_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths for data files (these will be in DATA_DIR after you place them)
MAHARASHTRA_DEM_PATH = os.path.join(DATA_DIR, 'maharashtra_dem.tif')
MAHARASHTRA_BOUNDARY_SHP_PATH = os.path.join(DATA_DIR, 'maharashtra_boundary.shp')

# Original downloaded file names (as seen in your image)
DOWNLOADED_DEM_TILES = [
    os.path.join(DATA_DIR, 'maharashtra_dem_tile1.tif.tif'),
    os.path.join(DATA_DIR, 'maharashtra_dem_tile2.tif.tif')
]
DOWNLOADED_GADM_SHP_BASE = os.path.join(DATA_DIR, 'gadm41_IND_1')

# Fallback/Generic Paths
DEM_ASC_PATH = os.path.join(DATA_DIR, 'dem.asc')
COUNTRIES_SHP_PATH = os.path.join(DATA_DIR, 'ne_10m_admin_0_countries.shp')
POPULATED_PLACES_SHP_PATH = os.path.join(DATA_DIR, 'ne_10m_populated_places.shp')
RIVERS_SHP_PATH = os.path.join(DATA_DIR, 'ne_10m_rivers_lake_centerlines.shp')


# --- Helper Functions (Same as before) ---
def unzip_all_input_files(input_dir: str, output_dir: str):
    # This function is more relevant for initial setup or if files are zipped on server.
    # For Streamlit, we assume files are already extracted into DATA_DIR.
    st.info(f"Checking for zip files in {input_dir} to unzip to {output_dir}...")
    found_zip = False
    if not os.path.exists(input_dir):
        st.warning(f"Input directory '{input_dir}' does not exist. Skipping unzipping.")
        return

    for item in os.listdir(input_dir):
        if item.endswith(".zip"):
            found_zip = True
            zip_path = os.path.join(input_dir, item)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    st.write(f"Attempting to extract {item}...")
                    zip_ref.extractall(output_dir)
                st.success(f"Successfully extracted {item} to {output_dir}")
            except zipfile.BadZipFile:
                st.error(f"Warning: {item} is not a valid zip file. Skipping.")
            except Exception as e:
                st.error(f"Error extracting {item}: {e}")
    if not found_zip:
        st.info(f"No .zip files found in {input_dir}. Assuming data is already extracted or provided directly.")


def _create_dummy_dem(path: str, xllcorner: float, yllcorner: float):
    """Creates a dummy DEM at a specified path."""
    if os.path.exists(path):
        st.info(f"DEM already exists at {path}.")
        return path
    st.info(f"Creating dummy DEM at: {path}")
    rows, cols = 100, 100
    cell_size = 0.5
    nodata_value = -9999
    dem_data = np.random.rand(rows, cols) * 500 + 10
    header = f"""ncols        {cols}
nrows        {rows}
xllcorner    {xllcorner}
yllcorner    {yllcorner}
cellsize     {cell_size}
NODATA_value {nodata_value}
"""
    with open(path, 'w') as f:
        f.write(header)
        np.savetxt(f, dem_data, fmt='%.2f')
    st.success(f"Created dummy DEM at: {path}")
    return path


def _create_dummy_countries_shp(path: str):
    """Creates a dummy 'ne_10m_admin_0_countries.shp' with an 'India' polygon."""
    if os.path.exists(path):
        st.info(f"Dummy countries shapefile already exists at {path}.")
        return path
    st.info(f"Creating dummy countries shapefile at: {path}")
    india_coords = [(70.0, 10.0), (70.0, 35.0), (95.0, 35.0), (95.0, 10.0), (70.0, 10.0)]
    india_polygon = Polygon(india_coords)
    gdf = gpd.GeoDataFrame({
        'ADMIN': ['India', 'Other Country'],
        'geometry': [india_polygon, Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])]
    }, crs="EPSG:4326")
    try:
        gdf.to_file(path)
        st.success(f"Created dummy countries shapefile at: {path}")
        return path
    except Exception as e:
        st.error(f"Error creating dummy countries shapefile: {e}")
        return None


def _create_dummy_populated_places_shp(path: str):
    """Creates a dummy 'ne_10m_populated_places.shp' with some Indian cities."""
    if os.path.exists(path):
        st.info(f"Dummy populated places shapefile already exists at {path}.")
        return path
    st.info(f"Creating dummy populated places shapefile at: {path}")
    data = {
        'NAME': ['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Nagpur'],
        'ADM0NAME': ['India', 'India', 'Karnataka', 'Maharashtra', 'Maharashtra'],
        'ADM1NAME': ['Maharashtra', 'Delhi', 'Karnataka', 'Maharashtra', 'Maharashtra'],
        'geometry': [
            Point(72.87, 19.07),  # Mumbai
            Point(77.20, 28.61),  # Delhi
            Point(77.59, 12.97),  # Bangalore
            Point(73.85, 18.52),  # Pune
            Point(79.08, 21.14)  # Nagpur
        ]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    try:
        gdf.to_file(path)
        st.success(f"Created dummy populated places shapefile at: {path}")
        return path
    except Exception as e:
        st.error(f"Error creating dummy populated places shapefile: {e}")
        return None


# --- Custom Data Preprocessing for Streamlit ---
def preprocess_downloaded_data():
    """
    Handles renaming and mosaicing of downloaded files to match expected paths.
    Assumes raw files are already in 'data/'.
    """
    st.subheader("Data Preprocessing Status")

    # 1. Handle Maharashtra Boundary Shapefile (gadm41_IND_1.* -> maharashtra_boundary.*)
    if os.path.exists(f"{DOWNLOADED_GADM_SHP_BASE}.shp") and not os.path.exists(MAHARASHTRA_BOUNDARY_SHP_PATH):
        st.info(f"Renaming GADM files to {os.path.basename(MAHARASHTRA_BOUNDARY_SHP_PATH)} base name.")
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj']:
            src_path = f"{DOWNLOADED_GADM_SHP_BASE}{ext}"
            dst_path = f"{os.path.splitext(MAHARASHTRA_BOUNDARY_SHP_PATH)[0]}{ext}"
            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dst_path)
                    st.write(f"Moved {os.path.basename(src_path)} to {os.path.basename(dst_path)}")
                except Exception as e:
                    st.error(f"Error moving {os.path.basename(src_path)}: {e}")
    elif os.path.exists(MAHARASHTRA_BOUNDARY_SHP_PATH):
        st.info(f"'{os.path.basename(MAHARASHTRA_BOUNDARY_SHP_PATH)}' already exists. Skipping GADM renaming.")
    else:
        st.warning(f"GADM files ({DOWNLOADED_GADM_SHP_BASE}.*) not found. Ensure they are in '{DATA_DIR}'.")

    # 2. Handle Maharashtra DEM Tiles (maharashtra_dem_tileX.tif.tif -> maharashtra_dem.tif)
    if not os.path.exists(MAHARASHTRA_DEM_PATH):
        existing_tiles = [p for p in DOWNLOADED_DEM_TILES if os.path.exists(p)]

        if existing_tiles:
            st.info(f"Found DEM tiles: {', '.join([os.path.basename(p) for p in existing_tiles])}")
            st.info(f"Mosaicing DEM tiles to create '{os.path.basename(MAHARASHTRA_DEM_PATH)}'.")

            src_files_to_mosaic = []
            for fp in existing_tiles:
                try:
                    src_files_to_mosaic.append(rasterio.open(fp))
                except Exception as e:
                    st.warning(f"Could not open DEM tile {os.path.basename(fp)}: {e}. Skipping this tile.")

            if src_files_to_mosaic:
                mosaic, out_transform = merge(src_files_to_mosaic)
                out_meta = src_files_to_mosaic[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                    "crs": src_files_to_mosaic[0].crs
                })
                with rasterio.open(MAHARASHTRA_DEM_PATH, "w", **out_meta) as dest:
                    dest.write(mosaic)
                st.success(f"Successfully mosaiced DEMs to {MAHARASHTRA_DEM_PATH}")
                for src in src_files_to_mosaic:
                    src.close()
                for fp in existing_tiles:
                    try:
                        os.remove(fp)
                        st.write(f"Removed original tile: {os.path.basename(fp)}")
                    except Exception as e:
                        st.error(f"Could not remove {os.path.basename(fp)}: {e}")
            else:
                st.warning("No valid DEM tiles found to mosaic. Ensure they are in '{DATA_DIR}' and not corrupted.")
        else:
            st.warning("No downloaded DEM tiles found. Ensure they are in '{DATA_DIR}'.")
    else:
        st.info(f"'{os.path.basename(MAHARASHTRA_DEM_PATH)}' already exists. Skipping DEM mosaicing.")

    st.success("Data Preprocessing Complete.")


# --- Helper Functions (Same as before, adapted for Streamlit output) ---
def _ensure_crs_match(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Ensures spatial layers have matching coordinate systems for analysis."""
    if gdf1.crs is None or gdf2.crs is None:
        st.warning(
            "Warning: One or both GeoDataFrames have undefined CRS. Attempting to proceed but results may be inaccurate.")
        return gdf1, gdf2
    if gdf1.crs != gdf2.crs:
        st.info(f"Reprojecting second GeoDataFrames from {gdf2.crs} to {gdf1.crs} for CRS match.")
        gdf2 = gdf2.to_crs(gdf1.crs)
    return gdf1, gdf2


def _convert_to_geotiff(input_path: str, output_path: str) -> str:
    """Converts raster formats to GeoTIFF for consistent processing"""
    with rasterio.open(input_path) as src:
        profile = src.profile
        profile.update(driver='GTiff')
        data = src.read()
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    return output_path


# --- Custom Geospatial Tool Functions (Adapted for Streamlit output) ---

def LoadRaster(input_path: str) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raster file not found: {input_path}")

    if input_path.lower().endswith('.asc'):
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(input_path).replace('.asc', '.tif'))
        st.info(f"Converting .asc to GeoTIFF: {input_path} -> {output_path}")
        return _convert_to_geotiff(input_path, output_path)

    try:
        with rasterio.open(input_path) as src:
            _ = src.profile
        if input_path.lower().endswith('.tif') and not input_path.startswith(OUTPUT_DIR):
            output_path_tif = os.path.join(OUTPUT_DIR, os.path.basename(input_path))
            if not os.path.exists(output_path_tif):
                shutil.copy(input_path, output_path_tif)
                st.info(f"Copied {input_path} to {output_path_tif}")
            return output_path_tif
        return input_path
    except Exception as e:
        raise Exception(f"Raster loading failed for {input_path}: {str(e)}")


def LoadVector(input_path: str) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Vector file not found: {input_path}")
    try:
        gdf = gpd.read_file(input_path)
        if gdf.empty:
            raise ValueError(f"Vector file {input_path} is empty or contains no valid geometries.")
        if input_path.lower().endswith('.shp') and not input_path.startswith(OUTPUT_DIR):
            output_path_shp = os.path.join(OUTPUT_DIR, os.path.basename(input_path))
            if not os.path.exists(output_path_shp):
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qpj']:
                    src_file = input_path.replace('.shp', ext)
                    dst_file = output_path_shp.replace('.shp', ext)
                    if os.path.exists(src_file):
                        shutil.copy(src_file, dst_file)
                st.info(f"Copied {input_path} and associated files to {output_path_shp}")
            return output_path_shp
        return input_path
    except Exception as e:
        raise Exception(f"Vector loading failed for {input_path}: {str(e)}")


def FilterToIndia(input_path: str) -> str:
    output_path = os.path.join(OUTPUT_DIR,
                               f"india_{os.path.basename(input_path).replace('.tif', '').replace('.shp', '')}.tif" if input_path.lower().endswith(
                                   '.tif') else f"india_{os.path.basename(input_path)}")

    india_boundary_path = os.path.join(OUTPUT_DIR, "india_boundary.shp")
    if not os.path.exists(india_boundary_path):
        try:
            countries = gpd.read_file(COUNTRIES_SHP_PATH)
            india = countries[countries['ADMIN'] == 'India']
            if india.empty:
                raise ValueError("India not found in countries dataset. Cannot filter to India.")
            india.to_file(india_boundary_path)
            st.info(f"Created India boundary shapefile at {india_boundary_path}")
        except Exception as e:
            raise Exception(f"Failed to create India boundary for filtering: {e}")

    if input_path.lower().endswith(('.tif', '.asc')):
        if "dem.tif" in os.path.basename(input_path):
            output_path = os.path.join(OUTPUT_DIR, "india_dem.tif")
        return ClipSpatialData(input_path, india_boundary_path, output_path)
    elif input_path.lower().endswith(('.shp', '.geojson')):
        gdf = gpd.read_file(input_path)
        india_gdf = gpd.read_file(india_boundary_path)
        gdf, india_gdf = _ensure_crs_match(gdf, india_gdf)

        filtered_gdf = gpd.GeoDataFrame()
        if 'ADMIN' in gdf.columns:
            filtered_gdf = gdf[gdf['ADMIN'].str.contains('India', case=False, na=False)]
        elif 'NAME' in gdf.columns:
            filtered_gdf = gdf[gdf['NAME'].str.contains('India', case=False, na=False)]

        if filtered_gdf.empty:
            st.warning(f"Attempting spatial join to filter {os.path.basename(input_path)} to India.")
            if gdf.geometry.name != 'geometry':
                gdf = gdf.set_geometry(gdf.geometry.name)
            joined_gdf = gpd.sjoin(gdf, india_gdf, how='inner', predicate='intersects')
            filtered_gdf = joined_gdf.drop(columns=['index_right'])

        if filtered_gdf.empty:
            st.warning(
                f"No features found within India for {os.path.basename(input_path)}. Outputting empty shapefile.")
            empty_gdf = gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
            empty_gdf.to_file(output_path)
            return output_path

        filtered_gdf.to_file(output_path)
        return output_path
    else:
        raise ValueError(f"Unsupported input file format for FilterToIndia: {os.path.splitext(input_path)[1]}")


def FilterToState(input_path: str, state_name: str) -> str:
    output_path = os.path.join(OUTPUT_DIR, f"{state_name.lower().replace(' ', '_')}_{os.path.basename(input_path)}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        gdf = gpd.read_file(input_path)
        filtered_gdf = gpd.GeoDataFrame()
        if 'ADM1NAME' in gdf.columns:
            filtered_gdf = gdf[gdf['ADM1NAME'].str.contains(state_name, case=False, na=False)]
        elif 'NAME_1' in gdf.columns:
            filtered_gdf = gdf[gdf['NAME_1'].str.contains(state_name, case=False, na=False)]

        if filtered_gdf.empty:
            st.warning(
                f"No features found for state '{state_name}' in {os.path.basename(input_path)}. Outputting empty shapefile.")
            empty_gdf = gpd.GeoDataFrame(columns=gdf.columns, geometry=[], crs=gdf.crs)
            empty_gdf.to_file(output_path)
            return output_path

        filtered_gdf.to_file(output_path)
        return output_path
    except Exception as e:
        raise Exception(f"State-level filtering failed for {input_path} to {state_name}: {str(e)}")


def CalculateFlowDirection(dem_path: str) -> str:
    output_path = os.path.join(OUTPUT_DIR, "flow_direction.tif")
    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            rows, cols = dem.shape
            flow_dir = np.zeros_like(dem, dtype=np.uint8)
            directions = [
                (0, 1, 1), (1, 1, 2), (1, 0, 4), (1, -1, 8),
                (0, -1, 16), (-1, -1, 32), (-1, 0, 64), (-1, 1, 128)
            ]
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if dem[r, c] == src.nodata:
                        continue
                    current_elevation = dem[r, c]
                    max_drop = -1
                    best_direction_code = 0
                    for dr, dc, code in directions:
                        nr, nc = r + dr, c + dc
                        neighbor_elevation = dem[nr, nc]
                        if neighbor_elevation != src.nodata:
                            drop = current_elevation - neighbor_elevation
                            if drop > max_drop:
                                max_drop = drop
                                best_direction_code = code
                    flow_dir[r, c] = best_direction_code
            profile = src.profile.copy()
            profile.update(dtype=np.uint8, count=1)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(flow_dir, 1)
        return output_path
    except Exception as e:
        raise Exception(f"Flow direction calculation failed for {dem_path}: {str(e)}")


def CalculateFlowAccumulation(flow_dir_path: str) -> str:
    output_path = os.path.join(OUTPUT_DIR, "flow_accumulation.tif")
    try:
        with rasterio.open(flow_dir_path) as src:
            flow_dir = src.read(1)
            rows, cols = flow_dir.shape
            flow_acc = np.ones((rows, cols), dtype=np.float64)
            flow_mapping = {
                1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
                16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
            }
            in_degree = np.zeros_like(flow_dir, dtype=int)
            for r in range(rows):
                for c in range(cols):
                    current_flow_code = flow_dir[r, c]
                    if current_flow_code == 0 or current_flow_code == src.nodata:
                        continue
                    if current_flow_code in flow_mapping:
                        dr_out, dc_out = flow_mapping[current_flow_code]
                        next_r, next_c = r + dr_out, c + dc_out
                        if 0 <= next_r < rows and 0 <= next_c < cols and flow_dir[next_r, next_c] != src.nodata:
                            in_degree[next_r, next_c] += 1
            queue = deque()
            for r in range(rows):
                for c in range(cols):
                    if in_degree[r, c] == 0 and flow_dir[r, c] != src.nodata:
                        queue.append((r, c))
            processed_cells = 0
            with st.spinner("Processing flow accumulation..."):
                while queue:
                    r, c = queue.popleft()
                    processed_cells += 1
                    if (processed_cells % 10000) == 0:
                        st.write(f"Processed {processed_cells} cells for flow accumulation...")
                    current_flow_code = flow_dir[r, c]
                    if current_flow_code == 0 or current_flow_code == src.nodata:
                        continue
                    if current_flow_code in flow_mapping:
                        dr_out, dc_out = flow_mapping[current_flow_code]
                        next_r, next_c = r + dr_out, c + dc_out
                        if 0 <= next_r < rows and 0 <= next_c < cols and flow_dir[next_r, next_c] != src.nodata:
                            flow_acc[next_r, next_c] += flow_acc[r, c]
                            in_degree[next_r, next_c] -= 1
                            if in_degree[next_r, next_c] == 0:
                                queue.append((next_r, next_c))
            st.info(f"Total cells processed for flow accumulation: {processed_cells}")
            profile = src.profile.copy()
            profile.update(dtype=np.float64, count=1)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(flow_acc, 1)
            return output_path
    except Exception as e:
        raise Exception(f"Flow accumulation failed for {flow_dir_path}: {str(e)}")


def ExtractStreamNetwork(flow_acc_path: str, threshold: float = 1000) -> str:
    output_path = os.path.join(OUTPUT_DIR, "stream_network.shp")
    try:
        with rasterio.open(flow_acc_path) as src:
            flow_acc = src.read(1)
            streams_raster = (flow_acc > threshold).astype(np.uint8)
            if src.nodata is not None:
                streams_raster[flow_acc == src.nodata] = 0
            extracted_shapes = (
                {'properties': {'value': v}, 'geometry': s}
                for s, v in shapes(streams_raster, mask=None, transform=src.transform)
            )
            gdf_polygons = gpd.GeoDataFrame.from_features(list(extracted_shapes), crs=src.crs)
            gdf_polygons = gdf_polygons[gdf_polygons['value'] == 1]
            lines = []
            for geom in gdf_polygons.geometry:
                if geom.geom_type == 'Polygon':
                    lines.append(geom.exterior)
                    lines.extend(geom.interiors)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        lines.append(poly.exterior)
                        lines.extend(poly.interiors)
                elif geom.geom_type == 'LineString':
                    lines.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    lines.extend(geom.geoms)
            if not lines:
                st.warning(f"No stream features found above threshold {threshold}. Outputting empty shapefile.")
                empty_gdf = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                empty_gdf.to_file(output_path)
                return output_path
            merged_lines = linemerge(lines)
            if merged_lines.geom_type == 'MultiLineString':
                final_gdf = gpd.GeoDataFrame(geometry=list(merged_lines.geoms), crs=src.crs)
            else:
                final_gdf = gpd.GeoDataFrame(geometry=[merged_lines], crs=src.crs)
            final_gdf.to_file(output_path)
            return output_path
    except Exception as e:
        raise Exception(f"Stream extraction failed for {flow_acc_path}: {str(e)}")


def DelineateWatershed(flow_dir_path: str,
                       pour_points: Union[str, Tuple[float, float], List[Tuple[float, float]]]) -> str:
    output_path = os.path.join(OUTPUT_DIR, "watershed.shp")
    try:
        with rasterio.open(flow_dir_path) as src:
            points_to_buffer = []
            if isinstance(pour_points, str):
                if not os.path.exists(pour_points):
                    raise FileNotFoundError(f"Pour point vector file not found: {pour_points}")
                pts_gdf = gpd.read_file(pour_points)
                if pts_gdf.empty:
                    raise ValueError(f"Pour point vector file {pour_points} is empty.")
                points_to_buffer.extend(pts_gdf.geometry)
            elif isinstance(pour_points, tuple) and len(pour_points) == 2:
                points_to_buffer.append(Point(pour_points))
            elif isinstance(pour_points, list) and all(isinstance(p, tuple) and len(p) == 2 for p in pour_points):
                points_to_buffer.extend([Point(p) for p in pour_points])
            else:
                raise ValueError(
                    "Invalid pour_points format. Must be a path to a vector file, a single (x,y) tuple, or a list of (x,y) tuples.")

            if not points_to_buffer:
                st.warning(
                    "No valid pour points found or provided for watershed delineation. Outputting empty shapefile.")
                empty_gdf = gpd.GeoDataFrame(geometry=[], crs=src.crs)
                empty_gdf.to_file(output_path)
                return output_path

            buffered_polygons = []
            buffer_distance_meters = 5000

            for pt in points_to_buffer:
                if src.crs and src.crs.is_geographic:
                    temp_gdf = gpd.GeoDataFrame(geometry=[pt], crs=src.crs)
                    temp_gdf_proj = temp_gdf.to_crs(epsg=3857)
                    buffered_point_proj = temp_gdf_proj.geometry.iloc[0].buffer(buffer_distance_meters)
                    buffered_point = buffered_point_proj.to_crs(src.crs)
                else:
                    buffered_point = pt.buffer(buffer_distance_meters)
                buffered_polygons.append(buffered_point)

            watershed_union = unary_union(buffered_polygons)
            gdf_watershed = gpd.GeoDataFrame(geometry=[watershed_union], crs=src.crs)
            gdf_watershed.to_file(output_path)

            st.info("Note: DelineateWatershed in this prototype uses a simplified buffer around pour points.")
            st.info(
                "For full hydrological watershed delineation, consider using dedicated hydrology libraries if precise boundaries are required.")

            return output_path
    except Exception as e:
        raise Exception(f"Watershed delineation failed for {flow_dir_path}: {str(e)}")


def FloodInundation(dem_path: str, water_level: float) -> str:
    output_path = os.path.join(OUTPUT_DIR, f"flood_{water_level}m.shp")
    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            flood_binary_raster = np.zeros_like(dem, dtype=np.uint8)
            if src.nodata is not None:
                valid_data_mask = (dem != src.nodata)
                flood_binary_raster[valid_data_mask] = (dem[valid_data_mask] <= water_level).astype(np.uint8)
            else:
                flood_binary_raster = (dem <= water_level).astype(np.uint8)
            features = []
            for s, v in shapes(flood_binary_raster, transform=src.transform):
                if v == 1:
                    features.append({'properties': {'water_level': water_level}, 'geometry': s})
            if not features:
                st.warning(f"No flood areas found at {water_level}m water level. Creating empty shapefile.")
                flood_gdf = gpd.GeoDataFrame(columns=['water_level'], geometry=[], crs=src.crs)
            else:
                flood_gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs)
            if 'water_level' in flood_gdf.columns and len('water_level') > 10:
                st.warning(f"Note: Column 'water_level' will be truncated to 'water_leve' when saving to shapefile.")
            flood_gdf.to_file(output_path)
            return output_path
    except Exception as e:
        raise Exception(f"Flood simulation failed for {dem_path}: {str(e)}")


def VisualizeFlood(flood_path: str, basemap_path: Optional[str] = None,
                   stream_network_path: Optional[str] = None) -> None:
    try:
        flood_gdf = gpd.read_file(flood_path)

        target_crs = CRS.from_epsg(3857)
        original_crs = flood_gdf.crs

        if original_crs is None:
            st.warning("Warning: Flood data has undefined CRS. Assuming EPSG:4326 and reprojecting to EPSG:3857.")
            flood_gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
            original_crs = flood_gdf.crs

        use_osm_basemap = False
        if (basemap_path is None or basemap_path == "") and flood_gdf.crs == target_crs:
            use_osm_basemap = True

        flood_gdf_proj = flood_gdf.to_crs(target_crs) if use_osm_basemap else flood_gdf

        stream_gdf_proj = None
        if stream_network_path and os.path.exists(stream_network_path):
            stream_gdf = gpd.read_file(stream_network_path)
            if stream_gdf.crs and stream_gdf.crs != target_crs:
                st.info(f"Reprojecting stream network from {stream_gdf.crs} to {target_crs} for visualization.")
                stream_gdf_proj = stream_gdf.to_crs(target_crs)
            else:
                stream_gdf_proj = stream_gdf

        fig, ax = plt.subplots(figsize=(12, 10))

        if use_osm_basemap:
            st.info("Adding OpenStreetMap basemap using Contextily.")
            try:
                if not flood_gdf_proj.empty:
                    minx, miny, maxx, maxy = flood_gdf_proj.total_bounds
                    buffer_x = (maxx - minx) * 0.1
                    buffer_y = (maxy - miny) * 0.1
                    extent = (minx - buffer_x, maxx + buffer_x, miny - buffer_y, maxy + buffer_y)
                elif stream_gdf_proj is not None and not stream_gdf_proj.empty:
                    minx, miny, maxx, maxy = stream_gdf_proj.total_bounds
                    buffer_x = (maxx - minx) * 0.1
                    buffer_y = (maxy - miny) * 0.1
                    extent = (minx - buffer_x, maxx + buffer_x, miny - buffer_y, maxy + buffer_y)
                else:
                    st.warning(
                        "Flood and stream data are empty, cannot determine extent for basemap. Skipping OSM basemap.")
                    extent = None

                if extent:
                    cx.add_basemap(ax, crs=target_crs, source=cx.providers.OpenStreetMap.Mapnik, zoom='auto')
                else:
                    st.warning("No valid extent for contextily basemap. Skipping.")

            except Exception as e:
                st.error(f"Error adding OpenStreetMap basemap: {e}. Falling back to no basemap.")
        elif basemap_path and os.path.exists(basemap_path):
            if basemap_path.lower().endswith(('.tif', '.asc')):
                with rasterio.open(basemap_path) as src:
                    basemap_data = src.read(1)
                    if src.nodata is not None:
                        basemap_data = np.where(basemap_data == src.nodata, np.nan, basemap_data)
                    im = ax.imshow(basemap_data,
                                   extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
                                   cmap='terrain', origin='upper', zorder=0)
                    plt.colorbar(im, ax=ax, label='Elevation (DEM Units)')
            elif basemap_path.lower().endswith(('.shp', '.geojson')):
                basemap_gdf = gpd.read_file(basemap_path)
                if basemap_gdf.crs and flood_gdf_proj.crs and basemap_gdf.crs != flood_gdf_proj.crs:
                    st.info(f"Reprojecting basemap from {basemap_gdf.crs} to {flood_gdf_proj.crs} for visualization.")
                    basemap_gdf = basemap_gdf.to_crs(flood_gdf_proj.crs)
                basemap_gdf.plot(ax=ax, color='lightgray', edgecolor='black', zorder=1)
        else:
            st.info("No basemap path provided and flood data is not in EPSG:3857. No basemap will be shown.")

        if stream_gdf_proj is not None and not stream_gdf_proj.empty:
            stream_gdf_proj.plot(ax=ax, color='darkblue', linewidth=1.0, label='Stream Network', zorder=3)
            plt.legend()

        if not flood_gdf_proj.empty:
            flood_gdf_proj.plot(ax=ax, color='blue', alpha=0.6, label='Flood Extent', zorder=4)
            plt.legend()
        else:
            st.warning("No flood data to plot.")

        combined_bounds = []
        if not flood_gdf_proj.empty:
            combined_bounds.append(flood_gdf_proj.total_bounds)
        if stream_gdf_proj is not None and not stream_gdf_proj.empty:
            combined_bounds.append(stream_gdf_proj.total_bounds)

        if combined_bounds:
            minx = min(b[0] for b in combined_bounds)
            miny = min(b[1] for b in combined_bounds)
            maxx = max(b[2] for b in combined_bounds)
            maxy = max(b[3] for b in combined_bounds)
            buffer_x = (maxx - minx) * 0.1
            buffer_y = (maxy - miny) * 0.1
            ax.set_xlim(minx - buffer_x, maxx + buffer_x)
            ax.set_ylim(miny - buffer_y, maxy + buffer_y)
        else:
            st.warning("No valid data to set map extent.")

        plot_title = "Flood Inundation and Stream Network Map"
        if not flood_gdf_proj.empty and 'water_level' in flood_gdf_proj.columns:
            water_levels = flood_gdf_proj['water_level'].unique()
            if len(water_levels) == 1:
                plot_title += f" (Water Level: {water_levels[0]:.1f}m)"

        ax.set_title(plot_title)
        ax.set_xlabel(f"Easting ({target_crs.to_string()})")
        ax.set_ylabel(f"Northing ({target_crs.to_string()})")
        plt.tight_layout()
        st.pyplot(fig)  # Display the plot in Streamlit
        plt.close(fig)  # Close the figure to free up memory
    except Exception as e:
        st.error(f"Visualization error for {os.path.basename(flood_path)}: {str(e)}")
        st.exception(e)  # Display full traceback in Streamlit


def VisualizeIndiaFlood(flood_path: str, state: Optional[str] = None, city: Optional[str] = None):
    # This function is not directly used in the advanced workflow example,
    # but kept for completeness if the LLM decides to use it.
    try:
        india_boundary_path = os.path.join(OUTPUT_DIR, "india_boundary.shp")
        if not os.path.exists(india_boundary_path):
            try:
                countries = gpd.read_file(COUNTRIES_SHP_PATH)
                india = countries[countries['ADMIN'] == 'India']
                if india.empty:
                    raise ValueError("India not found in countries dataset. Cannot visualize India flood.")
                india.to_file(india_boundary_path)
                st.info(f"Created India boundary shapefile at {india_boundary_path}")
            except Exception as e:
                raise Exception(f"Failed to create India boundary for visualization: {e}")

        flood_gdf = gpd.read_file(flood_path)
        india_gdf = gpd.read_file(india_boundary_path)

        target_crs = CRS.from_epsg(3857)

        if flood_gdf.crs and flood_gdf.crs != target_crs:
            flood_gdf = flood_gdf.to_crs(target_crs)
        elif flood_gdf.crs is None:
            st.warning("Warning: Flood data has undefined CRS. Assuming EPSG:4326 and reprojecting to EPSG:3857.")
            flood_gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
            flood_gdf = flood_gdf.to_crs(target_crs)

        if india_gdf.crs and india_gdf.crs != target_crs:
            india_gdf = india_gdf.to_crs(target_crs)
        elif india_gdf.crs is None:
            st.warning("Warning: India boundary has undefined CRS. Assuming EPSG:4326 and reprojecting to EPSG:3857.")
            india_gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
            india_gdf = india_gdf.to_crs(target_crs)

        fig, ax = plt.subplots(figsize=(12, 10))

        plot_title = "Flood Areas in India"
        focus_gdf = india_gdf

        if city:
            places = gpd.read_file(POPULATED_PLACES_SHP_PATH)
            if places.crs and places.crs != target_crs:
                places = places.to_crs(target_crs)
            city_place = places[places['NAME'].str.contains(city, case=False, na=False)]
            if not city_place.empty:
                city_area = city_place.geometry.iloc[0].buffer(20000)
                focus_gdf = gpd.GeoDataFrame(geometry=[city_area], crs=target_crs)
                plot_title = f"Flood Areas in {city}, India"
            else:
                st.warning(f"City '{city}' not found. Showing India-wide map.")

        elif state:
            places = gpd.read_file(POPULATED_PLACES_SHP_PATH)
            if places.crs and places.crs != target_crs:
                places = places.to_crs(target_crs)
            state_places = places[places['ADM1NAME'].str.contains(state, case=False, na=False)]
            if not state_places.empty:
                state_area = state_places.unary_union.convex_hull.buffer(50000)
                focus_gdf = gpd.GeoDataFrame(geometry=[state_area], crs=target_crs)
                plot_title = f"Flood Areas in {state}, India"
            else:
                st.warning(f"State '{state}' not found. Showing India-wide map.")

        if not focus_gdf.empty:
            minx, miny, maxx, maxy = focus_gdf.total_bounds
            cx.add_basemap(ax, crs=target_crs, source=cx.providers.OpenStreetMap.Mapnik, zoom='auto')
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
        else:
            st.warning("No valid focus area for basemap. Skipping OSM basemap.")

        if not india_gdf.empty:
            if focus_gdf is not india_gdf and not focus_gdf.empty:
                india_gdf_clipped = gpd.clip(india_gdf, focus_gdf)
                india_gdf_clipped.plot(ax=ax, color='none', edgecolor='black', zorder=1, linewidth=0.5)
            else:
                india_gdf.plot(ax=ax, color='none', edgecolor='black', zorder=1, linewidth=0.5)

        if not flood_gdf.empty:
            if focus_gdf is not india_gdf and not focus_gdf.empty:
                flood_gdf_clipped = gpd.clip(flood_gdf, focus_gdf)
                flood_gdf_clipped.plot(ax=ax, color='blue', alpha=0.6, label='Flood Extent', zorder=2)
            else:
                flood_gdf.plot(ax=ax, color='blue', alpha=0.6, label='Flood Extent', zorder=2)
            plt.legend()
        else:
            st.warning("No flood data to plot.")

        ax.set_title(plot_title)
        ax.set_xlabel(f"Easting (EPSG:3857)")
        ax.set_ylabel(f"Northing (EPSG:3857)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"India flood visualization failed: {str(e)}")
        st.exception(e)


def ClipSpatialData(input_path: str, clip_boundary_path: str, output_path: str) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data not found: {input_path}")
    if not os.path.exists(clip_boundary_path):
        raise FileNotFoundError(f"Clipping boundary not found: {clip_boundary_path}")

    try:
        clip_gdf = gpd.read_file(clip_boundary_path)
        if clip_gdf.empty:
            raise ValueError(f"Clipping boundary file {clip_boundary_path} is empty or invalid.")

        clip_geometry = clip_gdf.unary_union
        if not clip_geometry.is_valid:
            clip_geometry = clip_geometry.buffer(0)

        if input_path.lower().endswith(('.tif', '.asc')):
            with rasterio.open(input_path) as src:
                if src.crs and clip_gdf.crs and src.crs != clip_gdf.crs:
                    st.info(f"Reprojecting clipping boundary from {clip_gdf.crs} to raster CRS {src.crs}.")
                    clip_gdf = clip_gdf.to_crs(src.crs)
                    clip_geometry = clip_gdf.unary_union.buffer(0)

                out_image, out_transform = mask(src, [clip_geometry], crop=True, nodata=src.nodata)

                if out_image.ndim == 4 and out_image.shape[0] == 1:
                    out_image = out_image.squeeze(axis=0)

                if out_image.ndim == 2:
                    out_image = out_image[np.newaxis, :, :]

                if out_image.ndim != 3:
                    raise ValueError(
                        f"Final masked image has unexpected dimensions: {out_image.shape}. Expected (bands, height, width).")

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": src.crs,
                    "count": out_image.shape[0],
                    "nodata": src.nodata
                })

                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
            return output_path

        elif input_path.lower().endswith(('.shp', '.geojson')):
            input_gdf = gpd.read_file(input_path)
            input_gdf, clip_gdf_matched = _ensure_crs_match(input_gdf, clip_gdf)
            clipped_gdf = gpd.clip(input_gdf, clip_gdf_matched)
            if clipped_gdf.empty:
                st.warning(
                    f"Clipping resulted in an empty GeoDataFrame. No features from {input_path} intersected with {clip_boundary_path}.")
                empty_gdf = gpd.GeoDataFrame(columns=input_gdf.columns, geometry=[], crs=input_gdf.crs)
                empty_gdf.to_file(output_path)
                return output_path
            clipped_gdf.to_file(output_path)
            return output_path
        else:
            raise ValueError(f"Unsupported input file format for clipping: {os.path.splitext(input_path)[1]}")
    except Exception as e:
        raise Exception(f"Spatial clipping failed for {input_path} with {clip_boundary_path}: {str(e)}")


def CalculateAffectedPopulation(flood_extent_path: str, population_raster_path: str) -> float:
    if not os.path.exists(flood_extent_path):
        raise FileNotFoundError(f"Flood extent file not found: {flood_extent_path}")
    if not os.path.exists(population_raster_path):
        raise FileNotFoundError(f"Population raster file not found: {population_raster_path}")
    try:
        flood_gdf = gpd.read_file(flood_extent_path)
        if flood_gdf.empty:
            st.info("Flood extent is empty, so affected population is 0.")
            return 0.0
        with rasterio.open(population_raster_path) as pop_src:
            if pop_src.crs and flood_gdf.crs and flood_gdf.crs != pop_src.crs:
                st.info(f"Reprojecting flood extent from {flood_gdf.crs} to population raster CRS {pop_src.crs}.")
                flood_gdf = flood_gdf.to_crs(pop_src.crs)
            out_image, out_transform = mask(pop_src, flood_gdf.geometry, crop=True, nodata=pop_src.nodata)
            if pop_src.nodata is not None:
                affected_population = np.sum(out_image[out_image != pop_src.nodata])
            else:
                affected_population = np.sum(out_image)
            return float(affected_population)
    except Exception as e:
        raise Exception(f"Calculating affected population failed: {str(e)}")


def ReprojectSpatialData(input_path: str, target_crs_epsg: int, output_path: str) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data not found: {input_path}")
    try:
        target_crs = CRS.from_epsg(target_crs_epsg)
        if input_path.lower().endswith(('.tif', '.asc')):
            with rasterio.open(input_path) as src:
                if src.crs == target_crs:
                    st.info(
                        f"Raster {input_path} is already in target CRS EPSG:{target_crs_epsp}. Skipping reprojection.")
                    shutil.copy(input_path, output_path)
                    return output_path
                raster_dtype = src.read(1).dtype
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                destination_array = np.empty((height, width), dtype=raster_dtype)
                reproject(
                    source=rasterio.band(src, 1), destination=destination_array,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=target_crs,
                    resampling=Resampling.cubic, num_threads=os.cpu_count() or 1
                )
                profile = src.profile.copy()
                profile.update({
                    'crs': target_crs, 'transform': transform, 'width': width,
                    'height': height, 'dtype': raster_dtype
                })
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(destination_array, 1)
            return output_path
        elif input_path.lower().endswith(('.shp', '.geojson')):
            input_gdf = gpd.read_file(input_path)
            if input_gdf.crs == target_crs:
                st.info(f"Vector {input_path} is already in target CRS EPSG:{target_crs_epsg}. Skipping reprojection.")
                input_gdf.to_file(output_path)
                return output_path
            reprojected_gdf = input_gdf.to_crs(target_crs)
            reprojected_gdf.to_file(output_path)
            return output_path
        else:
            raise ValueError(f"Unsupported input file format for reprojection: {os.path.splitext(input_path)[1]}")
    except Exception as e:
        raise Exception(f"Spatial reprojection failed for {input_path} to EPSG:{target_crs_epsg}: {str(e)}")


def CalculateSlope(dem_path: str, output_slope_path: str) -> str:
    if not os.path.exists(dem_path):
        raise FileNotFoundError(f"Input DEM not found: {dem_path}")
    try:
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                dem_data[dem_data == src.nodata] = np.nan
            pixel_width = abs(src.transform[0])
            pixel_height = abs(src.transform[4])
            cell_size = (pixel_width + pixel_height) / 2.0
            if cell_size == 0:
                raise ValueError("Cannot calculate slope: Cell size is zero. Check DEM transform.")
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

            def gradient_x(neighborhood):
                if np.isnan(neighborhood).any(): return np.nan
                return np.sum(neighborhood.reshape(3, 3) * sobel_x) / (8 * cell_size)

            def gradient_y(neighborhood):
                if np.isnan(neighborhood).any(): return np.nan
                return np.sum(neighborhood.reshape(3, 3) * sobel_y) / (8 * cell_size)

            dz_dx = generic_filter(dem_data, gradient_x, size=3, mode='constant', cval=np.nan)
            dz_dy = generic_filter(dem_data, gradient_y, size=3, mode='constant', cval=np.nan)
            slope_radians = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
            slope_degrees = np.degrees(slope_radians)
            slope_degrees[np.isnan(slope_degrees)] = src.nodata if src.nodata is not None else 0
            profile = src.profile.copy()
            profile.update(dtype=np.float32, count=1, nodata=src.nodata)
            with rasterio.open(output_slope_path, 'w', **profile) as dst:
                dst.write(slope_degrees, 1)
        return output_path
    except Exception as e:
        raise Exception(f"Slope calculation failed for {dem_path}: {str(e)}")


# --- LLM Pipeline Setup ---
@st.cache_resource
def load_llm_pipeline():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm_pipeline = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
        },
        trust_remote_code=True,
        max_new_tokens=4000,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)


llm = load_llm_pipeline()

# --- Persona and Tool Context for Planning ---
persona_text_manual = """
You are an expert Geospatial AI Assistant specialized in flood management and spatial analysis.
Your role is to plan and execute GIS workflows from user queries using a set of powerful, internal Python tools.
Always respond in valid JSON wrapped in triple backtick blocks (```json).

**ABSOLUTELY CRITICAL RULES FOR PLAN GENERATION - READ CAREFULLY AND FOLLOW PRECISELY:**
1.  **STRICT SCHEMA ADHERENCE:** Your output JSON MUST validate against the `GeospatialPlan` schema provided.
2.  **MANDATORY FIELDS:** Every `step` in the `plan` list MUST have a **NON-EMPTY `reasoning` field**. Every `parameter` within a step MUST have a `type` field (e.g., "filepath", "float", "string", "boolean", "integer").
3.  **EXACT TOOL AND PARAMETER NAMES:** You **MUST ONLY USE** the tools listed below and their **EXACT parameter names**. DO NOT invent new tools, new parameter names, or alter existing ones.
    * For `LoadRaster` and `LoadVector`, use `input_path`.
    * For `FloodInundation`, use `dem_path` and `water_level`.
    * For `ClipSpatialData`, use `input_path`, `clip_boundary_path`, and `output_path`.
    * For `VisualizeFlood` and `VisualizeIndiaFlood`, use `flood_path` (and optionally `basemap_path`, `stream_network_path`, `state`, `city`).
    * For `CalculateFlowDirection`, use `dem_path`.
    * For `CalculateFlowAccumulation`, use `flow_dir_path`.
    * For `ExtractStreamNetwork`, use `flow_acc_path` and `threshold`.
    * For `DelineateWatershed`, use `flow_dir_path` and `pour_points`.
    * For `CalculateAffectedPopulation`, use `flood_extent_path` and `population_raster_path`.
    * For `ReprojectSpatialData`, use `input_path`, `target_crs_epsg`, and `output_path`.
    * For `CalculateSlope`, use `dem_path` and `output_slope_path`.
4.  **FILE PATHS:** All intermediate and final output file paths MUST be saved to the `flood_analysis_outputs/` directory. When referencing outputs from previous steps, use their full `flood_analysis_outputs/` path (e.g., `flood_analysis_outputs/my_output.shp`). **DO NOT use `[previous_output]` placeholders.**
5.  **OUTPUT TYPES:** Only use "vector", "raster", or "table" for `expected_outputs` types.

**AVAILABLE DATA FILES IN 'data/' DIRECTORY (after unzipping/creation):**
- `data/dem.asc` (Digital Elevation Model - **This is the raw DEM. It needs to be loaded by `LoadRaster` to create `flood_analysis_outputs/dem.tif` for other tools.**)
- `data/ne_10m_admin_0_countries.shp` (World Administrative Boundaries - Countries)
- `data/ne_10m_populated_places.shp` (World Populated Places)
- `data/ne_10m_rivers_lake_centerlines.shp` (World Rivers and Lake Centerlines)
- `data/maharashtra_dem.tif` (Higher-resolution DEM for Maharashtra - *User provided*)
- `data/maharashtra_boundary.shp` (Precise boundary for Maharashtra - *User provided*)

**PRIMARY RULE FOR DEM PROCESSING:**
- **IF ANY TOOL REQUIRES A DEM (e.g., `FloodInundation`, `CalculateFlowDirection`, `CalculateSlope`, `FilterToIndia` when filtering a DEM), YOU MUST ALWAYS START THE PLAN WITH `LoadRaster` USING THE MOST APPROPRIATE DEM. If a specific regional DEM like `data/maharashtra_dem.tif` is available and relevant to the query, use that. Otherwise, use `data/dem.asc` which will be converted to `flood_analysis_outputs/dem.tif`. NO EXCEPTIONS.**

Here are the specific tools you understand and their purposes:

1.  **LoadRaster(input_path: str)**: Loads and prepares raster data (e.g., Digital Elevation Models, land cover).
    * Purpose: To load gridded spatial data for analysis. **Crucially, this converts 'data/dem.asc' to 'flood_analysis_outputs/dem.tif' or copies user-provided .tif DEMs to 'flood_analysis_outputs/'.**
    * **Expected Parameters:** `input_path` (str) - **MUST be a path to an existing raster file, e.g., 'data/dem.asc' or 'data/maharashtra_dem.tif'**
    * Output: If converting .asc, outputs to `flood_analysis_outputs/dem.tif`. If copying a .tif, outputs to `flood_analysis_outputs/[original_filename].tif`. Otherwise, uses the input path.
    * Example: `LoadRaster("data/dem.asc")` or `LoadRaster("data/maharashtra_dem.tif")`

2.  **LoadVector(input_path: str)**: Loads and prepares vector data (e.g., shapefiles, GeoJSON for boundaries, points, lines).
    * Purpose: To load geographic features for analysis, visualization, or as boundaries.
    * **Expected Parameters:** `input_path` (str) - **MUST be a path to an existing vector file, e.g., 'data/ne_10m_admin_0_countries.shp' or 'data/maharashtra_boundary.shp'**
    * Output: Uses the input path directly, or copies to `flood_analysis_outputs/` if it's a .shp from `data/`.
    * Example: `LoadVector("data/ne_10m_admin_0_countries.shp")` (for country boundaries) or `LoadVector("data/maharashtra_boundary.shp")`

3.  **FilterToIndia(input_path: str)**: Filters a raster or vector dataset to only include features within the boundaries of India.
    * Purpose: To focus analysis specifically on India when requested by the user.
    * **Expected Parameters:** `input_path` (str) - Path to input file (raster or vector). **If `input_path` is `data/dem.asc`, ensure `LoadRaster("data/dem.asc")` is executed immediately before this step to create `flood_analysis_outputs/dem.tif`.**
    * Output: Path to filtered output in 'flood_analysis_outputs/india_[original_filename]'. For DEMs, specifically `flood_analysis_outputs/india_dem.tif`.
    * Example: `FilterToIndia("flood_analysis_outputs/dem.tif")` or `FilterToIndia("data/ne_10m_populated_places.shp")`

4.  **FilterToState(input_path: str, state_name: str)**: Filters a vector dataset to a specific Indian state.
    * Purpose: To narrow down analysis to a specific state within India. **The output of this tool can be used as a `clip_boundary_path` for `ClipSpatialData` to spatially filter other layers to the state.**
    * **Expected Parameters:** `input_path` (str) - Path to input vector file (e.g., 'data/ne_10m_populated_places.shp'), `state_name` (str) - The name of the Indian state (e.g., "Maharashtra").
    * Output: Path to filtered output in 'flood_analysis_outputs/[state]_[original_filename]'. This output can represent the approximate state boundary (e.g., 'flood_analysis_outputs/maharashtra_ne_10m_populated_places.shp').
    * Example: `FilterToState("data/ne_10m_populated_places.shp", "Maharashtra")`

5.  **CalculateFlowDirection(dem_path: str)**: Calculates the direction of water flow from each cell in a DEM.
    * Purpose: Essential for understanding water movement and as a prerequisite for other hydrological analyses.
    * **Expected Parameters:** `dem_path` (str) - **MUST be the path to the DEM output from `LoadRaster` or `FilterToIndia`, e.g., 'flood_analysis_outputs/dem.tif' or 'flood_analysis_outputs/india_dem.tif' or 'flood_analysis_outputs/maharashtra_dem.tif'**
    * Input: Expects a DEM raster.
    * Output: This tool consistently outputs to `flood_analysis_outputs/flow_direction.tif`.
    * Example: `CalculateFlowDirection("flood_analysis_outputs/dem.tif")`

6.  **CalculateFlowAccumulation(flow_dir_path: str)**: Computes the accumulated flow (number of upstream cells draining into each cell).
    * Purpose: Identifies stream channels and high-risk flood accumulation zones.
    * **Expected Parameters:** `flow_dir_path` (str) - **MUST be the path to the flow direction output from `CalculateFlowDirection`, e.g., 'flood_analysis_outputs/flow_direction.tif'**
    * Input: Expects the flow direction raster.
    * Output: This tool consistently outputs to `flood_analysis_outputs/flow_accumulation.tif`.
    * Example: `CalculateFlowAccumulation("flood_analysis_outputs/flow_direction.tif")`

7.  **ExtractStreamNetwork(flow_acc_path: str, threshold: float = 1000)**: Derives a vector network of streams from flow accumulation.
    * Purpose: To visualize the natural drainage system and potential flood propagation paths.
    * **Expected Parameters:** `flow_acc_path` (str), `threshold` (float)
    * Output: This tool consistently outputs to `flood_analysis_outputs/stream_network.shp`.
    * Example: `ExtractStreamNetwork("flood_analysis_outputs/flow_accumulation.tif", threshold=5000)`

8.  **DelineateWatershed(flow_dir_path: str, pour_points: Union[str, Tuple[float, float], List[Tuple[float, float]]])**: Delineates watershed boundaries upstream from specified points.
    * Purpose: To identify the contributing area for a specific location, useful for targeted flood planning.
    * **Expected Parameters:** `flow_dir_path` (str), `pour_points` (str or tuple or list of tuples)
    * Input: Expects the flow direction raster.
    * Output: This tool consistently outputs to `flood_analysis_outputs/watershed.shp`.
    * Example: `DelineateWatershed("flood_analysis_outputs/flow_direction.tif", "data/ne_10m_populated_places.shp")`

9.  **FloodInundation(dem_path: str, water_level: float)**: Simulates the extent of a flood by identifying all areas in a DEM
    * Purpose: To predict submerged areas under a given flood scenario and create flood hazard maps.
    * **Expected Parameters:** `dem_path` (str) - **MUST be the path to the DEM output from `LoadRaster` or `FilterToIndia`, e.g., 'flood_analysis_outputs/dem.tif' or 'flood_analysis_outputs/india_dem.tif' or 'flood_analysis_outputs/maharashtra_dem.tif'.**
    * Input: Expects the DEM raster.
    * Output: This tool consistently outputs to `flood_analysis_outputs/flood_{water_level}m.shp`.
    * Example: `FloodInundation("flood_analysis_outputs/dem.tif", 160.0)`

10. **VisualizeFlood(flood_path: str, basemap_path: Optional[str] = None, stream_network_path: Optional[str] = None)**: Generates a visual map of the flood inundation, optionally overlaid on a basemap and stream network.
    * Purpose: To present the results of flood simulation in an easily understandable
             graphical format, aiding in decision-making and communication.
    LLM Usage: The final step in a flood analysis workflow to show the user
               the outcome. The agent can generate this map when asked to
               "show the flood areas" or "visualize the impact."
    * **Expected Parameters:** `flood_path` (str), `basemap_path` (Optional[str]), `stream_network_path` (Optional[str])
    * Output: Displays a matplotlib plot.
    * Example: `VisualizeFlood("flood_analysis_outputs/flood_160.0m.shp", "data/ne_10m_admin_0_countries.shp")`

11. **VisualizeIndiaFlood(flood_path: str, state: Optional[str] = None, city: Optional[str] = None)**: Specialized visualization for India floods.
    * Purpose: To show flood areas specifically for India, optionally zoomed to state or city level.
    * **Expected Parameters:** `flood_path` (str), `state` (optional str), `city` (optional str).
    * Output: Displays a matplotlib plot focused on the requested area.
    * Example: `VisualizeIndiaFlood("flood_analysis_outputs/flood_160.0m.shp", city="Mumbai")`

12. **ClipSpatialData(input_path: str, clip_boundary_path: str, output_path: str)**: Clips a raster or vector dataset to the extent of a specified polygon boundary.
    * Purpose: To focus analysis on a specific area of interest (e.g., a country, a city) by removing data outside that boundary.
    * **Expected Parameters:** `input_path` (str), `clip_boundary_path` (str), `output_path` (str) - **`output_path` IS A MANDATORY PARAMETER AND MUST ALWAYS BE PROVIDED with a valid path in 'flood_analysis_outputs/' directory.**
    * Input: `clip_boundary_path` must be a path to a vector file (e.g., `data/ne_10m_admin_0_countries.shp`, `data/maharashtra_boundary.shp` or an output from `FilterToState`).
    * Output: This tool consistently outputs to the specified `output_path` within `flood_analysis_outputs/`.
    * Example: `ClipSpatialData("flood_analysis_outputs/flood_160.0m.shp", "data/ne_10m_admin_0_countries.shp", "flood_analysis_outputs/clipped_flood_countries.shp")`

13. **CalculateAffectedPopulation(flood_extent_path: str, population_raster_path: str)**: Estimates the total population affected by a flood by overlaying the flood extent with a population density raster.
    * Purpose: To quantify the human impact of a flood event for emergency response and damage assessment.
    * **Expected Parameters:** `flood_extent_path` (str), `population_raster_path` (str)
    * **WARNING: A suitable population raster file is NOT provided in the 'data/' directory. DO NOT use this tool unless the user explicitly provides a path to a population raster file (e.g., 'data/my_population_data.tif'). If you use this without a valid population raster, it WILL FAIL.**
    * Returns: float: The estimated total population affected.

14. **ReprojectSpatialData(input_path: str, target_crs_epsg: int, output_path: str)**: Transforms a spatial dataset from its current Coordinate Reference System (CRS) to a specified target CRS.
    * Purpose: To ensure consistency in CRSs across different datasets for accurate spatial operations.
    * **Expected Parameters:** `input_path` (str), `target_crs_epsg` (int), `output_path` (str)
    * Output: This tool consistently outputs to `flood_analysis_outputs/`.
    * Example: `ReprojectSpatialData("flood_analysis_outputs/dem.tif", 4326, "flood_analysis_outputs/dem_wgs84.tif")`

15. **CalculateSlope(dem_path: str, output_slope_path: str)**: Calculates the slope (steepness) of the terrain from a Digital Elevation Model (DEM).
    * Purpose: To identify steep areas prone to rapid runoff, erosion, or landslides, and flat areas more susceptible to pooling and flooding.
    * **Expected Parameters:** `dem_path` (str), `output_slope_path` (str)
    * Output: This tool consistently outputs to `flood_analysis_outputs/slope.tif`.
    * Example: `CalculateSlope("flood_analysis_outputs/dem.tif", "flood_analysis_outputs/slope.tif")`

When planning a workflow, consider the inputs and outputs of these tools and chain them logically to fulfill the user's request. Always save intermediate and final outputs to the `flood_analysis_outputs/` directory.
"""

persona_doc_manual = Document(
    page_content=persona_text_manual.strip(),
    metadata={"source": "Internal_GIS_Tools_Persona_Description"}
)

tool_doc_urls = []
tool_loader = WebBaseLoader(tool_doc_urls)
tool_docs_raw = tool_loader.load()


def sanitize(doc):
    clean = re.sub(r"```.*?```", "", doc.page_content, flags=re.DOTALL)
    clean = re.sub(r"{.*?}", "", clean)
    clean = re.sub(r"#+ ", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return Document(page_content=clean, metadata=doc.metadata)


tool_docs_clean = [sanitize(doc) for doc in tool_docs_raw]
all_documents = [persona_doc_manual] + tool_docs_clean
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_chunks = splitter.split_documents(all_documents)
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
main_vectorstore = FAISS.from_documents(docs_chunks, embedding_model)


class ToolParameter(BaseModel):
    name: str = Field(description="The name of the parameter (e.g., 'input_layer', 'distance').")
    value: Union[str, List[str]] = Field(
        description="The value(s) for the parameter (e.g., 'rivers.shp', '500 meters', or ['file1.tif', 'file2.tif']).")
    type: str = Field(
        description="The data type of the parameter (e.g., 'filepath', 'float', 'string', 'boolean', 'integer').")


class ExpectedOutput(BaseModel):
    name: str = Field(description="The suggested name for the output file (e.g., 'flood_risk_areas.shp').")
    type: Literal["vector", "raster", "table"] = Field(description="The type of geospatial output.")
    description: str = Field(description="A brief description of what this output represents.")


class GeoprocessingStep(BaseModel):
    step_number: int = Field(description="The sequential number of the step.")
    description: str = Field(description="A concise description of the geoprocessing operation.")
    tool: str = Field(
        description="The name of the custom geospatial tool function (e.g., 'LoadRaster', 'CalculateFlowAccumulation', 'ClipSpatialData').")
    parameters: List[ToolParameter] = Field(default_factory=list,
                                            description="A list of parameters required for the tool, including their names, values, and types.")
    reasoning: str = Field(
        description="Explain the Chain-of-Thought: why this specific step and tool are needed for the overall objective.")
    function: Optional[str] = Field(None,
                                    description="Optional: A specific function to apply within the tool (e.g., 'filter', 'dissolve').")
    arguments: Optional[Dict[str, Any]] = Field(None,
                                                description="Optional: Additional arguments for the function, as a dictionary.")


class GeospatialPlan(BaseModel):
    chain_of_thought_summary: str = Field(
        description="A high-level summary of the overall thought process and strategy for solving the geospatial problem. This should be concise and explain the 'why' behind the sequence of steps.")
    plan: List[GeoprocessingStep] = Field(
        description="A list of sequential geoprocessing steps to achieve the user's request.")
    expected_outputs: List[ExpectedOutput] = Field(
        description="A list of the final expected geospatial outputs from the plan.")


def extract_json_block(text: str) -> Optional[str]:
    json_wrapped_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_wrapped_match:
        return json_wrapped_match.group(1).strip()
    return None


schema_json = GeospatialPlan.model_json_schema()
schema_str = json.dumps(schema_json, indent=2)


def combine_documents(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# --- EXAMPLE_JSON for Plan Generation (Advanced Workflow) ---
EXAMPLE_JSON = '''{
  "chain_of_thought_summary": "To conduct an advanced flood-prone area analysis for Maharashtra, we will first process the DEM to derive hydrological features like flow direction, flow accumulation, and stream networks. Simultaneously, we will simulate flood inundation. Both the simulated flood areas and the derived stream networks will then be precisely clipped to the Maharashtra boundary and visualized together on an OpenStreetMap basemap.",
  "plan": [
    {
      "step_number": 1,
      "description": "Load and prepare the high-resolution Digital Elevation Model (DEM) for Maharashtra.",
      "tool": "LoadRaster",
      "parameters": [
        {
          "name": "input_path",
          "value": "data/maharashtra_dem.tif",
          "type": "filepath"
        }
      ],
      "reasoning": "The Maharashtra-specific DEM is the foundational input for all subsequent hydrological and flood inundation analyses, ensuring geographic precision."
    },
    {
      "step_number": 2,
      "description": "Calculate the direction of water flow from each cell in the Maharashtra DEM.",
      "tool": "CalculateFlowDirection",
      "parameters": [
        {
          "name": "dem_path",
          "value": "flood_analysis_outputs/maharashtra_dem.tif",
          "type": "filepath"
        }
      ],
      "reasoning": "Flow direction is a critical intermediate step for understanding water movement patterns and is required for calculating flow accumulation."
    },
    {
      "step_number": 3,
      "description": "Compute the accumulated flow, representing the number of upstream cells draining into each cell.",
      "tool": "CalculateFlowAccumulation",
      "parameters": [
        {
          "name": "flow_dir_path",
          "value": "flood_analysis_outputs/flow_direction.tif",
          "type": "filepath"
        }
      ],
      "reasoning": "Flow accumulation identifies areas where water naturally converges, which are potential stream channels and high-risk flood accumulation zones."
    },
    {
      "step_number": 4,
      "description": "Derive a vector network representing significant streams from the flow accumulation raster.",
      "tool": "ExtractStreamNetwork",
      "parameters": [
        {
          "name": "flow_acc_path",
          "value": "flood_analysis_outputs/flow_accumulation.tif",
          "type": "filepath"
        },
        {
          "name": "threshold",
          "value": "1000",
          "type": "float"
        }
      ],
      "reasoning": "Extracting the stream network provides a clear visualization of the drainage system, which is crucial for understanding flood pathways and adds complexity to the workflow."
    },
    {
      "step_number": 5,
      "description": "Simulate flood inundation at a representative water level (e.g., 160 meters) using the Maharashtra DEM.",
      "tool": "FloodInundation",
      "parameters": [
        {
          "name": "dem_path",
          "value": "flood_analysis_outputs/maharashtra_dem.tif",
          "type": "filepath"
        },
        {
          "name": "water_level",
          "value": "160.0",
          "type": "float"
        }
      ],
      "reasoning": "This step generates the primary flood extent polygon by identifying all areas below a simulated water surface, forming the core flood hazard layer."
    },
    {
      "step_number": 6,
      "description": "Load the precise Maharashtra administrative boundary shapefile.",
      "tool": "LoadVector",
      "parameters": [
        {
          "name": "input_path",
          "value": "data/maharashtra_boundary.shp",
          "type": "filepath"
        }
      ],
      "reasoning": "The precise Maharashtra boundary is essential for accurately clipping both the simulated flood extent and the extracted stream network to the region of interest."
    },
    {
      "step_number": 7,
      "description": "Clip the simulated flood extent to the precise boundaries of Maharashtra.",
      "tool": "ClipSpatialData",
      "parameters": [
        {
          "name": "input_path",
          "value": "flood_analysis_outputs/flood_160.0m.shp",
          "type": "filepath"
        },
        {
          "name": "clip_boundary_path",
          "value": "flood_analysis_outputs/maharashtra_boundary.shp",
          "type": "filepath"
        },
        {
          "name": "output_path",
          "value": "flood_analysis_outputs/maharashtra_clipped_flood.shp",
          "type": "filepath"
        }
      ],
      "reasoning": "Clipping the flood extent ensures that the final flood map is strictly confined to Maharashtra, making the results highly relevant and focused for the presentation."
    },
    {
      "step_number": 8,
      "description": "Clip the extracted stream network to the precise boundaries of Maharashtra.",
      "tool": "ClipSpatialData",
      "parameters": [
        {
          "name": "input_path",
          "value": "flood_analysis_outputs/stream_network.shp",
          "type": "filepath"
        },
        {
          "name": "clip_boundary_path",
          "value": "flood_analysis_outputs/maharashtra_boundary.shp",
          "type": "filepath"
        },
        {
          "name": "output_path",
          "value": "flood_analysis_outputs/maharashtra_clipped_stream_network.shp",
          "type": "filepath"
        }
      ],
      "reasoning": "Clipping the stream network to Maharashtra provides context for the flood areas and demonstrates comprehensive spatial processing within the region."
    },
    {
      "step_number": 9,
      "description": "Visualize the clipped flood-prone areas and the clipped stream network on an OpenStreetMap basemap.",
      "tool": "VisualizeFlood",
      "parameters": [
        {
          "name": "flood_path",
          "value": "flood_analysis_outputs/maharashtra_clipped_flood.shp",
          "type": "filepath"
        },
        {
          "name": "basemap_path",
          "value": "",
          "type": "string"
        },
        {
          "name": "stream_network_path",
          "value": "flood_analysis_outputs/maharashtra_clipped_stream_network.shp",
          "type": "filepath"
        }
      ],
      "reasoning": "The final visualization combines flood extent and stream networks on an interactive OpenStreetMap, providing a clear and impactful demonstration of the advanced GIS workflow for the hackathon presentation."
    }
  ],
  "expected_outputs": [
    {
      "name": "maharashtra_advanced_flood_map.png",
      "type": "raster",
      "description": "A high-quality visual map (PNG) displaying both the simulated flood extent and the stream network within Maharashtra on an OpenStreetMap basemap, demonstrating advanced GIS capabilities."
    },
    {
      "name": "maharashtra_clipped_flood.shp",
      "type": "vector",
      "description": "A shapefile containing the simulated flood-prone areas precisely clipped to the Maharashtra boundary."
    },
    {
      "name": "maharashtra_clipped_stream_network.shp",
      "type": "vector",
      "description": "A shapefile containing the extracted stream network precisely clipped to the Maharashtra boundary."
    }
  ]
}'''

# --- RAG Prompt Template (Same as before) ---
rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """
 You are an expert Geospatial AI Assistant.
 
 Your job is to convert user questions into step-by-step geospatial workflow plans.
 **YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT THAT STRICTLY ADHERES TO THE `GeospatialPlan` SCHEMA BELOW.**
 
 The top-level JSON object must contain these three keys:
 1.  **"chain_of_thought_summary"**: A 1-2 sentence explanation of your reasoning.
 2.  **"plan"**: A LIST of step-by-step tasks.
 3.  **"expected_outputs"**: A LIST of the final expected outputs.
 
 Each step within the "plan" list must have:
 -   step_number
 -   description
 -   tool
 -   parameters (all values must be strings)
 -   **reasoning (THIS FIELD IS ABSOLUTELY REQUIRED AND MUST NEVER BE EMPTY FOR ANY STEP)**
 
 Each output within the "expected_outputs" list must have:
 -   name
 -   type ("vector", "raster", or "table" only)
 -   description
 
 **ABSOLUTELY CRITICAL RULES FOR PLAN GENERATION - READ CAREFULLY AND FOLLOW PRECISELY:**
 1.  **STRICT SCHEMA ADHERENCE:** Your output JSON MUST validate against the `GeospatialPlan` schema provided.
 2.  **MANDATORY FIELDS:** Every `step` in the `plan` list MUST have a **NON-EMPTY `reasoning` field**. Every `parameter` within a step MUST have a `type` field (e.g., "filepath", "float", "string", "boolean", "integer").
 3.  **EXACT TOOL AND PARAMETER NAMES:** You **MUST ONLY USE** the tools listed below and their **EXACT parameter names**. DO NOT invent new tools, new parameter names, or alter existing ones.
     * For `LoadRaster` and `LoadVector`, use `input_path`.
     * For `FloodInundation`, use `dem_path` and `water_level`.
     * For `ClipSpatialData`, use `input_path`, `clip_boundary_path`, and `output_path`.
     * For `VisualizeFlood` and `VisualizeIndiaFlood`, use `flood_path` (and optionally `basemap_path`, `stream_network_path`, `state`, `city`).
     * For `CalculateFlowDirection`, use `dem_path`.
     * For `CalculateFlowAccumulation`, use `flow_dir_path`.
     * For `ExtractStreamNetwork`, use `flow_acc_path` and `threshold`.
     * For `DelineateWatershed`, use `flow_dir_path` and `pour_points`.
     * For `CalculateAffectedPopulation`, use `flood_extent_path` and `population_raster_path`.
     * For `ReprojectSpatialData`, use `input_path`, `target_crs_epsg`, and `output_path`.
     * For `CalculateSlope`, use `dem_path` and `output_slope_path`.
 4.  **FILE PATHS:** All intermediate and final output file paths MUST be saved to the `flood_analysis_outputs/` directory. When referencing outputs from previous steps, use their full `flood_analysis_outputs/` path (e.g., `flood_analysis_outputs/my_output.shp`). **DO NOT use `[previous_output]` placeholders.**
 5.  **OUTPUT TYPES:** Only use "vector", "raster", or "table" for `expected_outputs` types.
 
 **AVAILABLE DATA FILES IN 'data/' DIRECTORY (after unzipping/creation):**
 - `data/dem.asc` (Digital Elevation Model - **This is the raw DEM. It needs to be loaded by `LoadRaster` to create `flood_analysis_outputs/dem.tif` for other tools.**)
 - `data/ne_10m_admin_0_countries.shp` (World Administrative Boundaries - Countries)
 - `data/ne_10m_populated_places.shp` (World Populated Places)
 - `data/ne_10m_rivers_lake_centerlines.shp` (World Rivers and Lake Centerlines)
 - `data/maharashtra_dem.tif` (Higher-resolution DEM for Maharashtra - *User provided*)
 - `data/maharashtra_boundary.shp` (Precise boundary for Maharashtra - *User provided*)
 
 **PRIMARY RULE FOR DEM PROCESSING:**
 - **IF ANY TOOL REQUIRES A DEM (e.g., `FloodInundation`, `CalculateFlowDirection`, `CalculateSlope`, `FilterToIndia` when filtering a DEM), YOU MUST ALWAYS START THE PLAN WITH `LoadRaster` USING THE MOST APPROPRIATE DEM. If a specific regional DEM like `data/maharashtra_dem.tif` is available and relevant to the query, use that. Otherwise, use `data/dem.asc` which will be converted to `flood_analysis_outputs/dem.tif`. NO EXCEPTIONS.**
 
 SCHEMA:
 {json_schema_content}
 
 EXAMPLE OUTPUT:
 ```json
 {{example_json_content}}
 ```
 
 Use the following custom geospatial tools:
 {context}
 """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "User: {input}")
])

# --- Tool Execution Mapping (Same as before) ---
tool_mapping = {
    "LoadRaster": LoadRaster,
    "LoadVector": LoadVector,
    "FilterToIndia": FilterToIndia,
    "FilterToState": FilterToState,
    "CalculateFlowDirection": CalculateFlowDirection,
    "CalculateFlowAccumulation": CalculateFlowAccumulation,
    "ExtractStreamNetwork": ExtractStreamNetwork,
    "DelineateWatershed": DelineateWatershed,
    "FloodInundation": FloodInundation,
    "VisualizeFlood": VisualizeFlood,
    "VisualizeIndiaFlood": VisualizeIndiaFlood,
    "ClipSpatialData": ClipSpatialData,
    "CalculateAffectedPopulation": CalculateAffectedPopulation,
    "ReprojectSpatialData": ReprojectSpatialData,
    "CalculateSlope": CalculateSlope,
}


# --- Conversational Chain (Adapted for Streamlit) ---
def _get_context_for_query(query: str, retriever) -> str:
    retrieved_docs = retriever.invoke(query)
    return combine_documents(retrieved_docs)


def _execute_plan(plan_json: str) -> str:
    try:
        # --- Preprocess LLM-generated JSON to fix common errors before Pydantic validation ---
        try:
            raw_plan_dict = json.loads(plan_json)

            # Guardrail 1: Ensure 'reasoning' exists for every step
            if 'plan' in raw_plan_dict and isinstance(raw_plan_dict['plan'], list):
                for i, step in enumerate(raw_plan_dict['plan']):
                    if 'reasoning' not in step or not step['reasoning']:
                        step[
                            'reasoning'] = f"Auto-generated reasoning for step {i + 1} due to missing field in LLM output."
                        st.warning(f"GUARDRAIL: Injected missing 'reasoning' for step {i + 1}.")

                    # Guardrail 2: Ensure 'type' exists for every parameter
                    if 'parameters' in step and isinstance(step['parameters'], list):
                        for j, param in enumerate(step['parameters']):
                            if 'type' not in param or not param['type']:
                                if isinstance(param.get('value'), str):
                                    if param['value'].lower().endswith(('.shp', '.tif', '.asc', '.geojson')):
                                        param['type'] = 'filepath'
                                    elif re.match(r"^-?\d+\.\d+$", param['value']):
                                        param['type'] = 'float'
                                    elif re.match(r"^-?\d+$", param['value']):
                                        param['type'] = 'integer'
                                    else:
                                        param['type'] = 'string'
                                else:
                                    param['type'] = 'string'
                                st.warning(
                                    f"GUARDRAIL: Injected missing 'type' '{param['type']}' for parameter '{param.get('name', 'N/A')}' in step {i + 1}.")

            # Guardrail 3: Correct 'expected_outputs' types
            if 'expected_outputs' in raw_plan_dict and isinstance(raw_plan_dict['expected_outputs'], list):
                for output in raw_plan_dict['expected_outputs']:
                    if output.get('type') == 'image':
                        output['type'] = 'raster'
                        st.warning("GUARDRAIL: Corrected 'expected_outputs' type from 'image' to 'raster'.")

            plan_json_preprocessed = json.dumps(raw_plan_dict)

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse the generated plan JSON during preprocessing: {str(e)}")
            st.code(plan_json, language='json')
            return "Failed to parse generated plan."
        except Exception as e:
            st.error(f"An unexpected error occurred during JSON preprocessing: {str(e)}")
            st.exception(e)
            return "Preprocessing failed."

        # Now, validate the preprocessed JSON against Pydantic schema
        try:
            plan_data = GeospatialPlan.model_validate_json(plan_json_preprocessed)
        except ValidationError as ve:
            error_details = []
            for error in ve.errors():
                loc = ".".join(map(str, error['loc'])) if error['loc'] else "root"
                if error['type'] == 'missing':
                    error_details.append(f"Missing required field: '{loc}'")
                elif error['type'] == 'literal_error':
                    allowed = error['ctx'].get('expected', ['vector', 'raster', 'table'])
                    error_details.append(
                        f"Invalid value '{error['input']}' for field '{loc}'. "
                        f"Allowed values: {', '.join(map(str, allowed))}"
                    )
                elif error['type'] == 'string_type':
                    error_details.append(
                        f"Invalid type for field '{loc}'. Expected string, got {type(error['input']).__name__}.")
                else:
                    error_details.append(f"Validation error at '{loc}': {error['msg']}")

            st.error("Plan validation failed (after preprocessing). Please correct these issues:")
            for detail in error_details:
                st.markdown(f"- `{detail}`")
            st.info(
                "Common issues:\n- Every step MUST have a 'reasoning' field (though auto-injected if missing)\n- Output types must be 'vector', 'raster', or 'table' (though 'image' is corrected)\n- All parameter values must be strings")
            return "Plan validation failed."

        st.subheader("Executing Workflow Plan")
        outputs = {}

        # --- CRITICAL DEM PRE-CHECK GUARDRAIL ---
        dem_required_tools = ["FloodInundation", "CalculateFlowDirection", "CalculateSlope", "FilterToIndia"]

        dem_path_to_use = None
        if os.path.exists(MAHARASHTRA_DEM_PATH):
            dem_path_to_use = MAHARASHTRA_DEM_PATH
        else:
            dem_path_to_use = DEM_ASC_PATH

        needs_dem_guardrail = False
        for step in plan_data.plan:
            if step.tool in dem_required_tools:
                expected_output_dem = os.path.join(OUTPUT_DIR,
                                                   os.path.basename(dem_path_to_use).replace('.asc', '.tif'))
                if os.path.basename(expected_output_dem) not in outputs:
                    needs_dem_guardrail = True
                    break

        if needs_dem_guardrail:
            st.warning(
                f"GUARDRAIL: Detected that a DEM-dependent tool is planned, but '{os.path.basename(dem_path_to_use).replace('.asc', '.tif')}' does not exist in outputs.")
            st.info(f"GUARDRAIL: Forcing execution of LoadRaster('{dem_path_to_use}') as a prerequisite.")
            try:
                loaded_dem_path = LoadRaster(dem_path_to_use)
                outputs[os.path.basename(loaded_dem_path)] = loaded_dem_path
                st.success(f"GUARDRAIL: Successfully loaded and converted/copied DEM to {loaded_dem_path}")
            except Exception as e:
                st.error(
                    f"GUARDRAIL FAILED: Could not load initial DEM ('{dem_path_to_use}') required by the plan. Error: {str(e)}")
                st.exception(e)
                return "DEM loading failed."

        for step in plan_data.plan:
            tool_name = step.tool
            st.markdown(f"**Step {step.step_number}:** {step.description} (Tool: `{tool_name}`)")

            if tool_name not in tool_mapping:
                st.error(
                    f"Error: Tool '{tool_name}' not recognized. Please check the tool definitions. The LLM might be hallucinating tools.")
                return "Tool not recognized."

            tool_func = tool_mapping[tool_name]

            try:
                converted_params = {}
                for param_def in step.parameters:
                    original_name = param_def.name
                    param_value = param_def.value
                    param_type = param_def.type

                    corrected_name = original_name
                    if tool_name == "LoadRaster" and original_name == "input_layer":
                        corrected_name = "input_path"
                    elif tool_name == "LoadVector" and original_name == "input_layer":
                        corrected_name = "input_path"
                    elif tool_name == "ClipSpatialData":
                        if original_name == "input_layer":
                            corrected_name = "input_path"
                        elif original_name in ["clipping_layer", "clipping_geometry", "clipping_geometry_path"]:
                            corrected_name = "clip_boundary_path"
                        elif original_name == "output_layer":
                            corrected_name = "output_path"
                    elif tool_name in ["CalculateFlowDirection", "FloodInundation",
                                       "CalculateSlope"] and original_name == "input_layer":
                        corrected_name = "dem_path"
                    elif tool_name == "CalculateFlowAccumulation" and original_name == "input_path":
                        corrected_name = "flow_dir_path"
                    elif tool_name in ["VisualizeFlood", "VisualizeIndiaFlood"] and original_name == "input_path":
                        corrected_name = "flood_path"

                    if corrected_name != original_name:
                        st.info(
                            f"Corrected parameter name from '{original_name}' to '{corrected_name}' for tool '{tool_name}'.")

                    try:
                        if param_type == "float":
                            converted_value = float(param_value)
                        elif param_type == "int" or param_type == "integer":
                            converted_value = int(param_value)
                        elif param_type == "boolean":
                            converted_value = str(param_value).lower() == 'true'
                        elif param_type == "filepath":
                            if param_value.startswith("flood_analysis_outputs/"):
                                base_name = os.path.basename(param_value)
                                converted_value = outputs.get(base_name, param_value)
                            elif param_value.startswith("data/"):
                                data_file_map = {
                                    "data/dem.asc": DEM_ASC_PATH,
                                    "data/ne_10m_admin_0_countries.shp": COUNTRIES_SHP_PATH,
                                    "data/ne_10m_populated_places.shp": POPULATED_PLACES_SHP_PATH,
                                    "data/ne_10m_rivers_lake_centerlines.shp": RIVERS_SHP_PATH,
                                    "data/maharashtra_dem.tif": MAHARASHTRA_DEM_PATH,
                                    "data/maharashtra_boundary.shp": MAHARASHTRA_BOUNDARY_SHP_PATH
                                }
                                converted_value = data_file_map.get(param_value, os.path.join(DATA_DIR,
                                                                                              os.path.basename(
                                                                                                  param_value)))
                            else:
                                converted_value = param_value
                        else:
                            converted_value = param_value
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Failed to convert parameter '{corrected_name}' with value '{param_value}' "
                            f"to type '{param_type}': {str(e)}"
                        )

                    converted_params[corrected_name] = converted_value

                if tool_name == "FloodInundation":
                    if "dem_path" not in converted_params:
                        raise ValueError("FloodInundation requires a 'dem_path' parameter.")
                    if not os.path.exists(converted_params["dem_path"]):
                        raise FileNotFoundError(
                            f"DEM file for FloodInundation not found: {converted_params['dem_path']}")
                    if "water_level" not in converted_params:
                        st.warning("Warning: 'water_level' not provided for FloodInundation. Defaulting to 160.0m.")
                        converted_params["water_level"] = 160.0

                    if not converted_params["dem_path"].lower().endswith(('.tif', '.asc')):
                        raise ValueError(
                            f"FloodInundation 'dem_path' must be a raster (DEM), but got: {converted_params['dem_path']}")

                    if "output_path" not in converted_params or not converted_params["output_path"].startswith(
                            OUTPUT_DIR):
                        default_flood_output = os.path.join(OUTPUT_DIR, f"flood_{converted_params['water_level']}m.shp")
                        st.info(
                            f"GUARDRAIL: Setting default output_path for FloodInundation to: {default_flood_output}")
                        converted_params["output_path"] = default_flood_output

                if tool_name == "CalculateFlowDirection":
                    if "output_path" in converted_params:
                        st.info(f"GUARDRAIL: Removed unexpected 'output_path' parameter for {tool_name}.")
                        del converted_params["output_path"]

                if tool_name == "CalculateFlowAccumulation":
                    if "accumulation_threshold" in converted_params:
                        st.info(f"GUARDRAIL: Removed unexpected 'accumulation_threshold' parameter for {tool_name}.")
                        del converted_params["accumulation_threshold"]
                    if "output_path" in converted_params:
                        st.info(f"GUARDRAIL: Removed unexpected 'output_path' parameter for {tool_name}.")
                        del converted_params["output_path"]

                if tool_name == "ClipSpatialData" and "output_path" not in converted_params:
                    st.info("GUARDRAIL: 'output_path' missing for ClipSpatialData. Attempting to generate a default.")
                    input_base = os.path.basename(converted_params.get("input_path", "input"))
                    clip_base = os.path.basename(converted_params.get("clip_boundary_path", "clip"))
                    input_name_no_ext = os.path.splitext(input_base)[0]
                    input_ext = os.path.splitext(input_base)[1]
                    clip_name_no_ext = os.path.splitext(clip_base)[0]

                    default_output_path = os.path.join(OUTPUT_DIR,
                                                       f"clipped_{input_name_no_ext}_by_{clip_name_no_ext}{input_ext}")
                    st.info(f"GUARDRAIL: Setting default output_path for ClipSpatialData to: {default_output_path}")
                    converted_params["output_path"] = default_output_path

                if tool_name == "VisualizeIndiaFlood":
                    if "output_path" in converted_params:
                        st.info(f"GUARDRAIL: Removed unexpected 'output_path' parameter for {tool_name}.")
                        del converted_params["output_path"]

                st.info(f"Calling `{tool_name}` with parameters: `{converted_params}`")
                result = tool_func(**converted_params)
                st.success(f"Step {step.step_number} completed. Result: `{result}`")

                if result and isinstance(result, str) and os.path.exists(result):
                    output_key = os.path.basename(result)
                    outputs[output_key] = result
                    st.write(f"Stored file output: `{output_key}` -> `{result}`")
                elif result is not None:
                    outputs[f"step_{step.step_number}_{tool_name}_result"] = result
                    st.write(f"Stored non-file output for step {step.step_number} (`{tool_name}`): `{result}`")

            except FileNotFoundError as fnfe:
                st.error(f"Error in step {step.step_number}: File not found - {str(fnfe)}")
                st.exception(fnfe)
                return "File not found error."
            except Exception as e:
                st.error(
                    f"Error executing step {step.step_number}:\nTool: `{tool_name}`\nParameters: `{converted_params}`\nError: {str(e)}")
                st.exception(e)
                return "Execution failed."

        valid_output_types = {"vector", "raster", "table"}
        invalid_outputs = [
            out.name for out in plan_data.expected_outputs
            if out.type not in valid_output_types
        ]

        if invalid_outputs:
            st.warning("Workflow completed but with output type warnings:")
            for name in invalid_outputs:
                st.markdown(f"- Invalid output type for: `{name}`")
            st.info("Allowed types are: vector, raster, table. Please update your prompt to use correct output types.")

        st.success("Workflow completed successfully. All steps executed as planned.")
        return "Workflow completed successfully."

    except json.JSONDecodeError as jde:
        st.error(f"Failed to parse the generated plan JSON. Error details: {str(jde)}")
        st.code(plan_json, language='json')
        return "Failed to parse plan JSON."
    except Exception as e:
        st.error(f"An unexpected error occurred during plan execution: {str(e)}")
        st.exception(e)
        return "Unexpected error during plan execution."


# --- Performance Metrics Function ---
def measure_plan_runtime(plan_json_string: str, scenario_name: str) -> float:
    st.subheader(f"Measuring Runtime for Scenario: {scenario_name}")
    start_time = time.time()

    # Call the plan execution function
    execution_status = _execute_plan(plan_json_string)

    end_time = time.time()
    duration = end_time - start_time

    st.markdown(f"**Scenario '{scenario_name}' Completed.**")
    st.write(f"Execution Status: `{execution_status}`")
    st.write(f"Total Runtime: `{duration:.2f} seconds`")

    return duration


# --- Streamlit App Layout ---
def main():
    st.title("🗺️ Advanced Geospatial Flood Analysis Assistant")
    st.markdown("---")

    st.sidebar.header("Data Setup & Information")
    st.sidebar.info(
        "**Important:** For this demo, please ensure your `data/` directory "
        "is located in the same folder as this `app.py` file. "
        "It should contain: `gadm41_IND_1.*` (shapefile components) and "
        "`maharashtra_dem_tile1.tif.tif`, `maharashtra_dem_tile2.tif.tif`."
    )

    # Initialize India boundary file (moved here to ensure it's created once at startup)
    india_boundary_path_init = os.path.join(OUTPUT_DIR, "india_boundary.shp")
    if not os.path.exists(india_boundary_path_init):
        st.sidebar.info("Attempting to initialize India boundary shapefile...")
        try:
            countries_gdf = gpd.read_file(COUNTRIES_SHP_PATH)
            india = countries_gdf[countries_gdf['ADMIN'] == 'India']
            if not india.empty:
                india.to_file(india_boundary_path_init)
                st.sidebar.success(f"Successfully initialized India boundary shapefile at {india_boundary_path_init}")
            else:
                st.sidebar.warning(
                    f"Could not find 'India' in '{COUNTRIES_SHP_PATH}' to initialize india_boundary.shp. This might affect FilterToIndia and VisualizeIndiaFlood.")
        except Exception as e:
            st.sidebar.error(f"Error initializing india_boundary.shp: {e}")
            st.sidebar.exception(e)
            st.sidebar.warning(
                "Please ensure 'ne_10m_admin_0_countries.shp' is correctly extracted to the 'data/' directory.")

    # Preprocess data (renaming/mosaicing) on app startup
    with st.sidebar.expander("Run Data Preprocessing"):
        preprocess_downloaded_data()
        _create_dummy_dem(DEM_ASC_PATH, xllcorner=0.0, yllcorner=0.0)
        _create_dummy_countries_shp(COUNTRIES_SHP_PATH)
        _create_dummy_populated_places_shp(POPULATED_PLACES_SHP_PATH)
        st.success("Initial data checks and dummy data creation complete.")

    # --- Chat Interface ---
    st.header("Ask a Geospatial Question")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Type your geospatial query here...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            st.subheader("Generating Workflow Plan...")
            with st.spinner("Thinking..."):
                raw_llm_output = planner.invoke({
                    "input": user_query,
                    "chat_history": [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(
                        content=msg["content"]) for msg in st.session_state.chat_history],
                    "schema": schema_str
                })

            if raw_llm_output is None:
                st.error("Failed to generate a valid geospatial plan (no JSON block found). Please try rephrasing.")
                st.session_state.chat_history.append({"role": "assistant",
                                                      "content": "Failed to generate a valid geospatial plan (no JSON block found). Please try rephrasing."})
                return

            st.subheader("Chain of Thought")
            try:
                parsed_plan = json.loads(raw_llm_output)
                chain_of_thought = parsed_plan.get("chain_of_thought_summary", "No chain of thought summary provided.")
                st.info(chain_of_thought)
                st.subheader("Generated Workflow Plan (JSON)")
                st.code(json.dumps(parsed_plan, indent=2), language='json')
            except json.JSONDecodeError:
                st.error("Failed to parse the generated plan JSON for display.")
                st.code(raw_llm_output, language='json')
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": "Failed to parse the generated plan JSON for display."})
                return

            st.subheader("Executing Plan...")
            execution_placeholder = st.empty()  # Placeholder for execution messages
            with st.spinner("Executing geospatial operations..."):
                # Redirect prints from tools to Streamlit
                # This is a basic way; for more control, modify tools to take a logger or Streamlit object
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                redirected_output = StringIO()
                sys.stdout = redirected_output

                execution_result = _execute_plan(raw_llm_output)

                sys.stdout = old_stdout  # Restore stdout
                execution_placeholder.text_area("Execution Log", redirected_output.getvalue(), height=300)

            st.success(f"Assistant: {execution_result}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Plan executed: {execution_result}"})

    st.markdown("---")
    st.header("Performance Metrics (Example)")
    st.info(
        "These are hardcoded examples to demonstrate the metrics. For real comparison, you would define specific test plans.")

    # Hardcoded example plans for runtime comparison
    # IMPORTANT: These plans MUST be valid and runnable with your data setup.
    # Use the advanced workflow example for the "Maharashtra Flood Analysis"
    maharashtra_flood_plan_json_for_metrics = EXAMPLE_JSON  # Use the advanced example JSON

    # A simpler plan for comparison (e.g., just slope calculation)
    simple_slope_plan_json_for_metrics = """
{
  "chain_of_thought_summary": "To demonstrate a simpler process, we will load the generic DEM and calculate its slope.",
  "plan": [
    {
      "step_number": 1,
      "description": "Load the generic Digital Elevation Model (DEM).",
      "tool": "LoadRaster",
      "parameters": [
        {
          "name": "input_path",
          "value": "data/dem.asc",
          "type": "filepath"
        }
      ],
      "reasoning": "Loading the default DEM is the first step for any DEM-based analysis when no specific regional DEM is requested."
    },
    {
      "step_number": 2,
      "description": "Calculate the slope of the loaded DEM.",
      "tool": "CalculateSlope",
      "parameters": [
        {
          "name": "dem_path",
          "value": "flood_analysis_outputs/dem.tif",
          "type": "filepath"
        },
        {
          "name": "output_slope_path",
          "value": "flood_analysis_outputs/generic_slope.tif",
          "type": "filepath"
        }
      ],
      "reasoning": "Calculating slope helps understand terrain characteristics, which is a fundamental geospatial operation."
    }
  ],
  "expected_outputs": [
    {
      "name": "generic_slope.tif",
      "type": "raster",
      "description": "A raster map showing the slope of the generic DEM."
    }
  ]
}
    """

    if st.button("Run Performance Comparison"):
        st.session_state.runtimes = {}
        with st.spinner("Running performance tests... This might take a while."):
            st.session_state.runtimes["Maharashtra Flood Analysis"] = measure_plan_runtime(
                maharashtra_flood_plan_json_for_metrics, "Maharashtra Flood Analysis (Advanced)")
            st.session_state.runtimes["Generic DEM Slope Calculation"] = measure_plan_runtime(
                simple_slope_plan_json_for_metrics, "Generic DEM Slope Calculation")

        st.subheader("Comparative Runtime Results")
        if "runtimes" in st.session_state:
            for scenario, duration in st.session_state.runtimes.items():
                st.write(f"- **{scenario}:** `{duration:.2f} seconds`")
        st.success("Performance comparison complete!")


if __name__ == "__main__":
    main()
