import streamlit as st
import os
import datetime
from datetime import datetime, timedelta
import pathlib
from typing import Union, Optional, List, Dict, Any, Tuple
import numpy as np
import tempfile

st.set_page_config(layout="wide", page_title="Mosaic Builder", page_icon="🧩")

st.title("🧩 Sentinel-2 Mosaic Builder")

st.markdown("""
This tool helps you create and regularize mosaics from Sentinel-2 satellite imagery heavily based on the S2Mosaic package from Nicholas Wright available at https://github.com/DPIRD-DMA/S2Mosaic.

### Key Features:
- Create mosaics for specific grid areas and time ranges
- Generate a stack of mosaics at regular intervals
- Apply cloud coverage filters and advanced masking using OmniCloudMask (https://github.com/DPIRD-DMA/OmniCloudMask)
- Export to GeoTIFF or NetCDF formats
- Stack time series for analysis

You need the s2mosaic library installed to use this tool.
""")

# Check if s2mosaic is installed
try:
    import s2mosaic
    import geopandas as gpd
    import folium
    from streamlit_folium import folium_static
    from folium.plugins import Draw
    HAS_S2MOSAIC = True
except ImportError:
    HAS_S2MOSAIC = False
    st.warning("⚠️ The s2mosaic library is not installed. This interface will show all options but won't be able to process mosaics.")
    st.info("To install s2mosaic, run: `pip install s2mosaic`")
    st.info("For the map selection feature, you'll also need: `pip install geopandas folium streamlit-folium`")

# Try to load the Sentinel-2 grid
s2_grid = None
if HAS_S2MOSAIC:
    try:
        # Get the path to the grid file in the s2mosaic package
        import pkg_resources
        grid_path = pkg_resources.resource_filename('s2mosaic', 'S2_grid/sentinel_2_index.gpkg')
        s2_grid = gpd.read_file(grid_path)
        HAS_GRID = True
    except Exception as e:
        st.warning(f"⚠️ Could not load Sentinel-2 grid: {str(e)}")
        HAS_GRID = False
else:
    HAS_GRID = False

# Function to create a simplified map for selecting grid cells
def create_grid_selection_map(grid_data, simplification_tolerance=0.2):
    """Create an optimized map for selecting grid cells"""
    
    # Apply stronger simplification to improve performance
    if grid_data.crs != "EPSG:4326":
        grid_data = grid_data.to_crs("EPSG:4326")
    
    # Apply stronger simplification
    grid_data_simple = grid_data.copy()
    grid_data_simple['geometry'] = grid_data_simple['geometry'].simplify(simplification_tolerance)
    
    # Create the map
    m = folium.Map(location=[45, 10], zoom_start=3)
    
    # Add lightweight GeoJSON
    gjson = folium.GeoJson(
        grid_data_simple,
        name='Sentinel-2 Grid',
        style_function=lambda x: {'fillColor': '#3186cc', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.2},
        tooltip=folium.GeoJsonTooltip(
            fields=['Name'],
            aliases=['Grid ID:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    )
    
    # Add to map
    gjson.add_to(m)
    
    # Add layer control with minimal options
    folium.LayerControl().add_to(m)
    return m

def create_preview_map(grid_ids, grid_data):
    """Create a preview map of selected grid tiles"""
    
    # Filter grid data to selected IDs
    selected_grids = grid_data[grid_data['Name'].isin(grid_ids)]
    
    if selected_grids.empty:
        return None
    
    # Make sure it's in WGS84
    if selected_grids.crs != "EPSG:4326":
        selected_grids = selected_grids.to_crs("EPSG:4326")
    
    # Calculate bounds of all selected grids
    bounds = selected_grids.total_bounds
    center_y = (bounds[1] + bounds[3]) / 2
    center_x = (bounds[0] + bounds[2]) / 2
    
    # Create map
    preview_map = folium.Map(location=[center_y, center_x], zoom_start=6)
    
    # Add each grid with a different color
    for idx, (_, grid) in enumerate(selected_grids.iterrows()):
        # Cycle through colors
        colors = ['#3186cc', '#cc3131', '#31cc31', '#cc31cc', '#cccc31']
        color = colors[idx % len(colors)]
        
        folium.GeoJson(
            grid.geometry,
            name=f'Grid {grid["Name"]}',
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0.4
            },
            tooltip=f'Grid ID: {grid["Name"]}'
        ).add_to(preview_map)
    
    # Add basemap options with proper attribution
    folium.TileLayer(
        'OpenStreetMap',
        attr='© OpenStreetMap contributors'
    ).add_to(preview_map)
    
    folium.TileLayer(
        'https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.jpg',
        name='Stamen Terrain',
        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under CC BY SA.'
    ).add_to(preview_map)
    
    folium.TileLayer(
        'https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}.png',
        name='Stamen Toner',
        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under CC BY SA.'
    ).add_to(preview_map)
    
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        name='Esri Satellite',
        attr='Tiles & Images © Esri'
    ).add_to(preview_map)
    
    # Add layer control
    folium.LayerControl().add_to(preview_map)
    
    return preview_map


# Function to find intersecting grid cells with a vector file
def find_intersecting_grids(vector_file, grid_data):
    """Find all grid cells that intersect with a vector file"""
    try:
        # Handle different file types
        file_extension = os.path.splitext(vector_file)[1].lower()
        
        # For shapefiles, we need to check if we have a proper shapefile
        if file_extension == '.shp':
            # Check if the SHX file exists
            if not os.path.exists(vector_file.replace('.shp', '.shx')):
                st.warning("Uploaded shapefile is missing its .shx component. Try uploading as a GeoJSON instead.")
                # Try to read it anyway, might work with some drivers
                user_aoi = gpd.read_file(vector_file, engine="pyogrio")
            else:
                user_aoi = gpd.read_file(vector_file)
        else:
            # For GeoJSON, KML, etc.
            user_aoi = gpd.read_file(vector_file)
        
        # Ensure the AOI is in the same CRS as the grid
        if user_aoi.crs != grid_data.crs:
            user_aoi = user_aoi.to_crs(grid_data.crs)
        
        # Find intersections
        intersects = []
        for _, grid_row in grid_data.iterrows():
            if any(user_aoi.geometry.intersects(grid_row.geometry)):
                intersects.append(grid_row['Name'])
        
        return intersects
    except Exception as e:
        st.error(f"Error processing vector file: {str(e)}")
        st.info("For shapefiles, you need to upload all related files (.shp, .shx, .dbf). Consider converting to GeoJSON format first.")
        return []

# Sidebar for inputs
with st.sidebar:
    st.header("Mosaic Settings")
    
    # Basic settings - provide option for manual input or vector file
    grid_selection_method = st.radio(
        "Grid Selection Method",
        ["Manual Input", "Vector File", "Bounding Box"],
        help="Choose how to select the Sentinel-2 grid tile(s). Vector File and Bounding Box will mask to the specific region."
    )
    


    if grid_selection_method == "Manual Input":
        if HAS_GRID:
            # Get sorted list of all available grid IDs from the gpkg
            grid_ids_list = sorted(s2_grid['Name'].unique().tolist())
            
            # Provide multiselect with all available grid IDs
            grid_ids = st.multiselect(
                "Grid ID(s) (Sentinel-2 tiles)",
                options=grid_ids_list,
                default=['40TFN'] if '40TFN' in grid_ids_list else [grid_ids_list[0]] if grid_ids_list else [],
                help="Select one or more grid IDs from the available Sentinel-2 tiles"
            )
        else:
            # Fallback to text input if grid data is not available
            grid_ids_input = st.text_input(
                "Grid ID(s) (Sentinel-2 tiles)", 
                "40TFN",
                help="Enter one or more grid IDs separated by commas (e.g., '40TFN, 40TGN'). Grid validation unavailable."
            )
            # Parse comma-separated list
            grid_ids = [id.strip() for id in grid_ids_input.split(',')]

    if grid_selection_method == "Vector File":
        st.info("""
        Upload a vector file to automatically select grid tiles:
        - For shapefiles, please ZIP all components (.shp, .shx, .dbf, etc.) and upload the ZIP file
        - GeoJSON (.geojson) files can be uploaded directly
        """)
        
        uploaded_file = st.file_uploader(
            "Upload vector file", 
            type=["zip", "geojson", "gpkg", "kml", "fgb"], 
            help="For shapefiles, upload a ZIP containing all component files"
        )
    if grid_selection_method == "Bounding Box":
        st.info("Define a bounding box to automatically select overlapping grid tiles")
        # We'll create a map interface for this

    # Time Range Settings
    st.subheader("Time Range Settings")
    
    # Date selection
    start_date = st.date_input(
        "Start Date", 
        datetime.now() - timedelta(days=30),
        help="Start date for mosaic generation"
    )
    
    # Option for single mosaic or time series
    mosaic_type = st.radio(
        "Mosaic Type",
        ["Single Mosaic", "Time Series"],
        help="Create a single mosaic or a series of mosaics at regular intervals"
    )
    
    if mosaic_type == "Single Mosaic":
        # Settings for single mosaic
        duration_days = st.slider(
            "Data Collection Duration (days)", 
            min_value=1, 
            max_value=60, 
            value=20,
            help="Number of days to collect data for the mosaic"
        )
        end_date = start_date
    else:
        # Settings for time series
        end_date = st.date_input(
            "End Date", 
            datetime.now(),
            help="End date for mosaic generation"
        )
        
        interval_type = st.selectbox(
            "Interval Type",
            ["day", "week", "month", "year"],
            index=1,  # default to week
            help="Type of interval between mosaics"
        )
        
        interval_value = st.number_input(
            f"Interval Value (how many {interval_type}s)", 
            min_value=1, 
            max_value=100, 
            value=1,
            help=f"Number of {interval_type}s between each mosaic"
        )
        
        interval_duration_days = st.slider(
            "Data Collection Duration (days)", 
            min_value=1, 
            max_value=60, 
            value=20,
            help="Number of days to collect data for each interval's mosaic"
        )
    
    # Output settings
    st.subheader("Output Settings")
    
    output_dir = st.text_input(
        "Output Directory", 
        "mosaic_output",
        help="Directory where output files will be saved"
    )
    
    # Stacking options for time series
    if mosaic_type == "Time Series":
        stack_output = st.checkbox(
            "Stack Outputs", 
            value=True,
            help="Combine all mosaics into a single multi-band GeoTIFF or NetCDF"
        )
        
        if stack_output:
            stack_format = st.radio(
                "Stack Format",
                ["NetCDF", "Multi-band GeoTIFF"],
                help="Format to use for stacked output"
            )
            
            # Set default stack filename based on selected grid(s)
            default_grid = "multiple_grids" if 'grid_ids' in locals() and len(grid_ids) > 1 else \
                          (grid_ids[0] if 'grid_ids' in locals() and len(grid_ids) > 0 else "40TFN")
            
            stack_filename = st.text_input(
                "Stack Filename", 
                f"{default_grid}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_stack",
                help="Name for the stacked output file (without extension)"
            )
    
    # Advanced options
    st.subheader("Advanced Settings")
    
    with st.expander("Mosaic Creation Options"):
        sort_method = st.selectbox(
            "Scene Sorting Method",
            ["valid_data", "oldest", "newest"],
            index=0,
            help="Method to sort scenes before mosaic creation"
        )
        
        mosaic_method = st.selectbox(
            "Mosaic Method",
            ["mean", "first"],
            index=0,
            help="Method to create the mosaic: 'mean' averages overlapping pixels, 'first' takes the first valid pixel"
        )
        
        required_bands = st.multiselect(
            "Required Bands",
            ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
            default=["B04", "B03", "B02", "B08"],
            help="Spectral bands to include (B04=Red, B03=Green, B02=Blue, B08=NIR)"
        )
        
        no_data_threshold = st.slider(
            "No Data Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.01, 
            step=0.01,
            help="Threshold for no data values (lower = more strict)"
        )
        
        overwrite = st.checkbox(
            "Overwrite Existing Files", 
            value=True,
            help="Whether to overwrite existing output files"
        )
        
        debug_cache = st.checkbox(
            "Debug Cache", 
            value=False,
            help="Cache downloads for faster debugging"
        )
    
    with st.expander("OCM Settings"):
        ocm_batch_size = st.slider(
            "OCM Batch Size", 
            min_value=1, 
            max_value=16, 
            value=1,
            help="Batch size for OCM inference"
        )
        
        ocm_inference_dtype = st.selectbox(
            "OCM Inference Data Type",
            ["bf16", "fp16", "fp32"],
            index=0,
            help="Data type for OCM inference"
        )
    
    with st.expander("Query Settings"):
        # Cloud cover filter
        cloud_cover_threshold = st.slider(
            "Cloud Cover Threshold (%)", 
            min_value=0, 
            max_value=100, 
            value=50,
            help="Maximum cloud cover percentage for scene selection"
        )
        
        # Additional filters (more advanced)
        include_cloud_shadow = st.checkbox(
            "Filter Cloud Shadow", 
            value=False,
            help="Include cloud shadow filter in query"
        )
        
        if include_cloud_shadow:
            cloud_shadow_threshold = st.slider(
                "Cloud Shadow Threshold (%)", 
                min_value=0, 
                max_value=100, 
                value=30,
                help="Maximum cloud shadow percentage for scene selection"
            )
    
    # Create a dictionary of additional query parameters
    additional_query = {'eo:cloud_cover': {'lt': cloud_cover_threshold}}
    
    if include_cloud_shadow:
        additional_query['eo:cloud_shadow'] = {'lt': cloud_shadow_threshold}

    
    # Start processing button
    process_button = st.button("Generate Mosaic", type="primary")

# Handle grid selection based on method
selected_grid_ids = []

# Handle grid selection based on method
selected_grid_ids = []

# Bounding Box Selection
if grid_selection_method == "Bounding Box" and HAS_GRID:
    st.subheader("Select Area of Interest")
    
    # Create columns for coordinate inputs
    col1, col2 = st.columns(2)
    with col1:
        min_lat = st.number_input("South Latitude", value=40.0, min_value=-90.0, max_value=90.0)
        min_lon = st.number_input("West Longitude", value=10.0, min_value=-180.0, max_value=180.0)
    
    with col2:
        max_lat = st.number_input("North Latitude", value=45.0, min_value=-90.0, max_value=90.0)
        max_lon = st.number_input("East Longitude", value=15.0, min_value=-180.0, max_value=180.0)
    
    # Create bounding box
    from shapely.geometry import box
    bbox = box(min_lon, min_lat, max_lon, max_lat)
    
    # Create a simple map to show the bounding box
    import folium
    from streamlit_folium import folium_static
    
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    bbox_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    # Add the bounding box
    folium.GeoJson(
        bbox,
        name='Bounding Box',
        style_function=lambda x: {
            'fillColor': '#3186cc',
            'color': '#000000',
            'weight': 2,
            'fillOpacity': 0.2
        }
    ).add_to(bbox_map)
    
    # Add proper tile layer with attribution
    folium.TileLayer(
        'OpenStreetMap',
        attr='© OpenStreetMap contributors'
    ).add_to(bbox_map)
    
    # Display the map
    folium_static(bbox_map, width=800, height=400)
    
    # Find intersecting grids
    with st.spinner("Finding intersecting grid tiles..."):
        # Convert grid data to same CRS as bbox (WGS84)
        grid_wgs84 = s2_grid.to_crs("EPSG:4326")
        
        # Find intersections
        intersecting_grids = []
        for _, grid_row in grid_wgs84.iterrows():
            if grid_row.geometry.intersects(bbox):
                intersecting_grids.append(grid_row['Name'])
        
        if intersecting_grids:
            st.success(f"Found {len(intersecting_grids)} intersecting grid tiles!")
            selected_grid_ids = st.multiselect(
                "Selected Grid Tiles",
                options=intersecting_grids,
                default=intersecting_grids,
                help="These grid tiles intersect with your bounding box"
            )
        else:
            st.warning("No intersecting grid tiles found. Please adjust your bounding box.")
            # Allow manual selection as fallback
            all_grid_ids = sorted(s2_grid['Name'].unique().tolist())
            selected_grid_ids = st.multiselect(
                "Select Grid ID(s) manually",
                all_grid_ids,
                default=['40TFN'] if '40TFN' in all_grid_ids else []
            )

# Vector File Selection
if grid_selection_method == "Vector File" and HAS_GRID and uploaded_file is not None:
    st.subheader("Grid Tiles from Vector File")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process the uploaded file based on type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.zip':
            # Save zip file to temp location
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Extract zip file
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Look for .shp file in the extracted contents
            shp_files = glob.glob(os.path.join(temp_dir, "*.shp"))
            if shp_files:
                vector_path = shp_files[0]
            else:
                st.error("No shapefile found in the ZIP file.")
                vector_path = None
        else:
            # For non-zip files, save directly
            vector_path = os.path.join(temp_dir, f"upload{file_extension}")
            with open(vector_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        
        # Process the vector file if we have a path
        if vector_path:
            with st.spinner("Finding intersecting grid tiles..."):
                # Read the vector file
                user_aoi = gpd.read_file(vector_path)
                
                # Ensure the AOI is in the same CRS as the grid
                if user_aoi.crs != s2_grid.crs:
                    user_aoi = user_aoi.to_crs(s2_grid.crs)
                
                # Find intersections
                intersecting_grids = []
                for _, grid_row in s2_grid.iterrows():
                    if any(user_aoi.geometry.intersects(grid_row.geometry)):
                        intersecting_grids.append(grid_row['Name'])
                
                if intersecting_grids:
                    st.success(f"Found {len(intersecting_grids)} intersecting grid tiles!")
                    selected_grid_ids = st.multiselect(
                        "Selected Grid Tiles",
                        options=intersecting_grids,
                        default=intersecting_grids,
                        help="These grid tiles intersect with your vector file"
                    )
                else:
                    st.warning("No intersecting grid tiles found. Please check your vector file.")
                    # Fallback to manual selection
                    all_grid_ids = sorted(s2_grid['Name'].unique().tolist())
                    selected_grid_ids = st.multiselect(
                        "Select Grid ID(s) manually",
                        all_grid_ids,
                        default=['40TFN'] if '40TFN' in all_grid_ids else []
                    )
    
    except Exception as e:
        st.error(f"Error processing vector file: {str(e)}")
        # Fallback to manual selection
        all_grid_ids = sorted(s2_grid['Name'].unique().tolist())
        selected_grid_ids = st.multiselect(
            "Select Grid ID(s) manually",
            all_grid_ids,
            default=['40TFN'] if '40TFN' in all_grid_ids else []
        )
    
    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

# Manual Input Selection
elif grid_selection_method == "Manual Input":
    selected_grid_ids = grid_ids

# Ensure we have at least one grid ID
if not selected_grid_ids:
    st.warning("Please select at least one grid tile.")
    selected_grid_ids = ["40TFN"]  # Default fallback

# Show preview of selected grids
if HAS_GRID and selected_grid_ids:
    st.subheader("Preview of Selected Grid Tiles")
    
    # Check if all selected grids exist
    valid_grids = [grid_id for grid_id in selected_grid_ids 
                   if grid_id in s2_grid['Name'].values]
    
    invalid_grids = [grid_id for grid_id in selected_grid_ids 
                    if grid_id not in s2_grid['Name'].values]
    
    if invalid_grids:
        st.warning(f"The following grid IDs were not found: {', '.join(invalid_grids)}")
    
    if valid_grids:
        # Create and display preview map
        preview_map = create_preview_map(valid_grids, s2_grid)
        if preview_map:
            folium_static(preview_map, width=800, height=400)
        
        # Show selected grid details
        st.subheader("Selected Grid Details")
        grid_details = []
        for grid_id in valid_grids:
            grid_info = s2_grid[s2_grid['Name'] == grid_id]
            if not grid_info.empty:
                # Create a condensed representation of grid details
                details = {
                    'Grid ID': grid_id,
                    'UTM Zone': grid_id[:2],
                    'Lat Band': grid_id[2],
                    'Grid Square': grid_id[3:],
                    'Area (sq deg)': f"{grid_info.geometry.area.iloc[0]:.2f}"
                }
                grid_details.append(details)
        
        # Display as a table
        if grid_details:
            st.table(grid_details)
    else:
        st.error("None of the specified grid IDs were found. Please check your input.")
        st.image("https://via.placeholder.com/800x400?text=No+Valid+Grid+IDs", use_column_width=True)

# Main content area - display configuration summary
st.subheader("Mosaic Configuration Summary")

col1, col2 = st.columns(2)

with col1:
    st.write("**Basic Settings**")
    st.write(f"- Grid IDs: {', '.join(selected_grid_ids)}")
    st.write(f"- Start Date: {start_date}")
    
    if mosaic_type == "Single Mosaic":
        st.write(f"- Duration: {duration_days} days")
    else:
        st.write(f"- End Date: {end_date}")
        st.write(f"- Interval: Every {interval_value} {interval_type}(s)")
        st.write(f"- Interval Duration: {interval_duration_days} days")
    
    st.write("**Output Settings**")
    st.write(f"- Output Directory: {output_dir}")
    
    if mosaic_type == "Time Series" and stack_output:
        st.write(f"- Stack Format: {stack_format}")
        st.write(f"- Stack Filename: {stack_filename}")

with col2:
    st.write("**Mosaic Settings**")
    st.write(f"- Sort Method: {sort_method}")
    st.write(f"- Mosaic Method: {mosaic_method}")
    st.write(f"- Required Bands: {', '.join(required_bands)}")
    
    st.write("**Query Filters**")
    st.write(f"- Cloud Cover: < {cloud_cover_threshold}%")
    
    if include_cloud_shadow:
        st.write(f"- Cloud Shadow: < {cloud_shadow_threshold}%")

# Processing logic
if process_button:
    st.subheader("Processing Mosaic")
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if not HAS_S2MOSAIC:
        st.error("Cannot process mosaic: s2mosaic library is not installed.")
        st.stop()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have valid grid IDs to process
    if not valid_grids:
        st.error("No valid grid IDs to process. Please check your selection.")
        st.stop()
    
    # Prepare masking parameters based on selection method
    mask_geometry = None
    mask_path = None
    bbox_coords = None
    
    if grid_selection_method == "Vector File" and uploaded_file is not None:
        # For vector file selection, use the uploaded file for masking
        temp_dir = tempfile.mkdtemp()
        try:
            # Save the uploaded vector file to a temporary location
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == '.zip':
                zip_path = os.path.join(temp_dir, "upload.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Extract zip file
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Look for .shp file in the extracted contents
                import glob
                shp_files = glob.glob(os.path.join(temp_dir, "*.shp"))
                if shp_files:
                    mask_path = shp_files[0]
            else:
                # For non-zip files, save directly
                mask_path = os.path.join(temp_dir, f"upload{file_extension}")
                with open(mask_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
        except Exception as e:
            st.warning(f"Error processing mask file: {str(e)}. Mosaics will not be masked to the vector boundary.")
    
    elif grid_selection_method == "Bounding Box":
        # For bounding box selection, use the bbox coordinates for masking
        bbox_coords = (min_lon, min_lat, max_lon, max_lat)
        st.info(f"Mosaics will be masked to the bounding box: {bbox_coords}")
    
    # Process each grid
    all_results = []
    
    for i, grid_id in enumerate(valid_grids):
        status_text.text(f"Processing grid {i+1}/{len(valid_grids)}: {grid_id}")
        progress_fraction = i / len(valid_grids)
        progress_bar.progress(progress_fraction)
        
        grid_output_dir = os.path.join(output_dir, grid_id) if len(valid_grids) > 1 else output_dir
        os.makedirs(grid_output_dir, exist_ok=True)
        
        if mosaic_type == "Single Mosaic":
            # Process single mosaic
            try:
                # Import and call the mosaic function for a single date range
                from s2mosaic import mosaic
                
                with st.spinner(f"Creating mosaic for grid {grid_id}..."):
                    status_text.text(f"Searching for scenes for grid {grid_id}...")
                    sub_progress = i / len(valid_grids) + (0.1 / len(valid_grids))
                    progress_bar.progress(sub_progress)
                    
                    status_text.text(f"Downloading scenes for grid {grid_id}...")
                    sub_progress = i / len(valid_grids) + (0.3 / len(valid_grids))
                    progress_bar.progress(sub_progress)
                    
                    status_text.text(f"Building mosaic for grid {grid_id}...")
                    sub_progress = i / len(valid_grids) + (0.7 / len(valid_grids))
                    progress_bar.progress(sub_progress)
                    
                    result = mosaic(
                        grid_id=grid_id,
                        start_year=start_date.year,
                        start_month=start_date.month,
                        start_day=start_date.day,
                        duration_days=duration_days,
                        output_dir=grid_output_dir,
                        sort_method=sort_method,
                        mosaic_method=mosaic_method,  # This controls how overlapping areas are merged ('mean' = average)
                        required_bands=required_bands,
                        no_data_threshold=no_data_threshold,
                        overwrite=overwrite,
                        ocm_batch_size=ocm_batch_size,
                        ocm_inference_dtype=ocm_inference_dtype,
                        debug_cache=debug_cache,
                        additional_query=additional_query
                    )
                    
                    # Apply masking if needed (for single mosaic)
                    if (mask_path or bbox_coords) and isinstance(result, pathlib.Path):
                        try:
                            import rasterio
                            from rasterio.mask import mask
                            
                            # Prepare geometry for masking
                            mask_geom = None
                            if bbox_coords:
                                # Create box geometry from bounding box
                                from shapely.geometry import box
                                mask_geom = [box(*bbox_coords).__geo_interface__]
                            elif mask_path:
                                # Read geometry from vector file
                                import geopandas as gpd
                                mask_gdf = gpd.read_file(mask_path)
                                mask_geom = mask_gdf.geometry.__geo_interface__
                            
                            # Apply mask if we have a geometry
                            if mask_geom:
                                with rasterio.open(result) as src:
                                    masked_data, masked_transform = mask(src, mask_geom, crop=True)
                                    
                                    # Create new metadata for masked file
                                    masked_meta = src.meta.copy()
                                    masked_meta.update({
                                        "height": masked_data.shape[1],
                                        "width": masked_data.shape[2],
                                        "transform": masked_transform
                                    })
                                    
                                    # Write masked file
                                    masked_path = os.path.splitext(result)[0] + "_masked.tif"
                                    with rasterio.open(masked_path, "w", **masked_meta) as dest:
                                        dest.write(masked_data)
                                    
                                    # Update result with masked file path
                                    result = masked_path
                        except Exception as e:
                            st.warning(f"Error applying mask: {str(e)}. Using unmasked output.")
                    
                    all_results.append(result)
                    st.success(f"Successfully created mosaic for grid {grid_id}")
                    
            except Exception as e:
                st.error(f"Error creating mosaic for grid {grid_id}: {str(e)}")
        
        else:
            # Process time series
            try:
                # Import the enhanced create_regularized_mosaics function
                from regularized_mosaic import create_regularized_mosaics
                
                # Convert date inputs to datetime objects for processing
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.min.time())
                
                # Create grid-specific stack filename if multiple grids
                grid_stack_filename = f"{grid_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_stack" \
                                      if len(valid_grids) > 1 else stack_filename
                
                with st.spinner(f"Creating time series for grid {grid_id}..."):
                    # Create regularized mosaics with masking if needed
                    results = create_regularized_mosaics(
                        grid_id=grid_id,
                        start_date=start_datetime,
                        end_date=end_datetime,
                        interval_type=interval_type,
                        interval_value=interval_value,
                        interval_duration_days=interval_duration_days,
                        output_dir=grid_output_dir,
                        sort_method=sort_method,
                        mosaic_method=mosaic_method,  # This controls how overlapping areas are merged ('mean' = average)
                        required_bands=required_bands,
                        no_data_threshold=no_data_threshold,
                        overwrite=overwrite,
                        ocm_batch_size=ocm_batch_size,
                        ocm_inference_dtype=ocm_inference_dtype,
                        debug_cache=debug_cache,
                        additional_query=additional_query,
                        stack_output=stack_output,
                        stack_filename=grid_stack_filename,
                        skip_existing=False,
                        # Add masking parameters
                        mask_path=mask_path,
                        bbox=bbox_coords
                    )
                    
                    all_results.extend(results)
                    st.success(f"Successfully created time series for grid {grid_id}")
                    
            except Exception as e:
                st.error(f"Error creating time series for grid {grid_id}: {str(e)}")
                st.exception(e)
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Show overall success message
    if all_results:
        if len(valid_grids) > 1:
            st.success(f"Processing complete! Generated mosaics for {len(valid_grids)} grid tiles. Results saved to: {output_dir}")
        else:
            st.success(f"Processing complete! Results saved to: {output_dir}")
    else:
        st.warning("No mosaics were created. Check your settings and try again.")
    
    # Cleanup temp directory if it was created
    if grid_selection_method == "Vector File" and 'temp_dir' in locals():
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

# Documentation section
with st.expander("About Sentinel-2 Mosaics"):
    st.markdown("""
    ### What is a Sentinel-2 Mosaic?
    
    A Sentinel-2 mosaic is a composite image created by combining multiple satellite scenes covering the same area over a specified time period. This helps to:
    - Fill in gaps due to clouds or sensor issues
    - Create a more complete view of an area
    - Reduce noise and artifacts in the data
    
    ### Grid IDs
    
    Sentinel-2 data is organized in a grid system using the Military Grid Reference System (MGRS). Each grid cell is identified by a code like "40TFN".
    
    - The first two digits (e.g., "40") represent the UTM zone
    - The letter (e.g., "T") represents the MGRS latitude band
    - The final two letters (e.g., "FN") identify the 100,000-meter grid square within the zone
    
    ### Common Band Combinations
    
    - **True Color**: B04 (Red), B03 (Green), B02 (Blue)
    - **False Color Infrared**: B08 (NIR), B04 (Red), B03 (Green)
    - **Agriculture**: B11 (SWIR), B08 (NIR), B02 (Blue)
    - **NDVI Components**: B08 (NIR), B04 (Red)
    
    ### Tips for Best Results
    
    - Use a longer duration window in cloudy regions to ensure enough clear views
    - Filter by cloud cover percentage to improve quality
    - The "valid_data" sort method often produces the best results
    - Use "mean" mosaic method for smooth results, "first" for less processing
    - When working with multiple grid tiles, consider memory limitations
    - For very large areas, process grid tiles individually and then merge the results
    """)

with st.expander("Working with Multiple Grid Tiles"):
    st.markdown("""
    ### Tips for Multi-Tile Processing
    
    When working with multiple Sentinel-2 grid tiles:
    
    1. **Output Organization**
       - When multiple tiles are selected, each tile's output is saved in a subdirectory named with the grid ID
       - This helps keep your files organized and prevents naming conflicts
    
    2. **Memory Considerations**
       - Processing multiple tiles requires more memory and disk space
       - Consider increasing resampling factor for large areas
       - For very large regions (10+ tiles), you might need to process in batches
    
    3. **Time Series Stacking**
       - Each grid gets its own time series stack when multiple grids are selected
       - The stacks use grid-specific filenames to avoid conflicts
    
    4. **Vector File Selection**
       - When using a vector file to select grids, all intersecting grids are included
       - You can manually remove any unwanted grids from the selection
       - For irregular areas, this approach is more efficient than manual selection
    
    5. **Merging Results**
       - After processing, you may want to merge tiles for a seamless mosaic
       - This can be done using GIS software or the `gdal_merge` tool
       - Example: `gdal_merge.py -o merged.tif input1.tif input2.tif`
    """)