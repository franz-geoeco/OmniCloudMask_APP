import os
import glob
import re
import rasterio
from collections import defaultdict

def find_raster_files(folder_path):
    """
    Find all raster files in a folder, ensuring no duplicates
    """
    extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    files = set()  # Using a set to avoid duplicates
    
    for ext in extensions:
        files.update(glob.glob(os.path.join(folder_path, f'*{ext}')))
    
    return list(files)


def is_multiband_file(file_path):
    """
    Check if a file is a multiband raster
    
    Parameters:
    file_path (str): Path to the raster file
    
    Returns:
    bool: True if the file has multiple bands, False otherwise
    """
    try:
        with rasterio.open(file_path) as src:
            return src.count > 1
    except Exception as e:
        print(f"Warning: Could not read file {os.path.basename(file_path)}: {str(e)}")
        return False


def group_files_by_date_tile(files):
    """
    Group files by date and tile ID if they follow naming pattern
    """
    grouped = defaultdict(set)  # Use a set instead of a list
    
    # Try different patterns for grouping
    patterns = [
        # Sentinel-2 style: SENTINEL-2_MSI_40TFN_B04_2022-05-13.tif
        r'.*_(\d{4}-\d{2}-\d{2})\.tif',
        # More generic date pattern: anything with YYYY-MM-DD
        r'.*(\d{4}-\d{2}-\d{2}).*\.tif',
        # Landsat style: LC08_L1TP_037029_20200628_20200708_01_T1_B4.TIF
        r'.*_(\d{8})_.*\.tif'
    ]
    
    # Track which files have been successfully matched
    matched_files = set()
    
    for file in files:
        if file in matched_files:
            continue  # Skip already matched files
            
        filename = os.path.basename(file)
        matched = False
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Format YYYYMMDD to YYYY-MM-DD if needed
                if len(date_str) == 8 and '-' not in date_str:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                
                grouped[date_str].add(file)
                matched_files.add(file)
                matched = True
                break  # Stop after first match
        
        if not matched and file not in matched_files:
            # If no date pattern found, try to group by other identifiers
            # but only if not already matched
            band_match = re.search(r'_B(\d+)_', filename, re.IGNORECASE)
            if band_match:
                tile_match = re.search(r'_(\w+)_B\d+_', filename, re.IGNORECASE)
                if tile_match:
                    tile_id = tile_match.group(1)
                    grouped[f"tile_{tile_id}"].add(file)
                    matched_files.add(file)
                else:
                    grouped["unknown"].add(file)
                    matched_files.add(file)
            else:
                grouped["unknown"].add(file)
                matched_files.add(file)
    
    # Convert sets back to lists for compatibility
    return {k: list(v) for k, v in grouped.items()}


def get_band_info(file_path):
    """
    Get information about bands in a raster file
    
    Parameters:
    file_path (str): Path to the raster file
    
    Returns:
    dict: Dictionary with band information
    """
    with rasterio.open(file_path) as src:
        band_info = {
            'count': src.count,
            'dtypes': src.dtypes,
            'nodata': src.nodata,
            'width': src.width,
            'height': src.height,
            'crs': src.crs,
            'transform': src.transform
        }
        
        # Try to get statistics for preview
        band_stats = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            # Skip no data values
            valid_data = band[band != src.nodata] if src.nodata is not None else band
            if len(valid_data) > 0:
                stats = {
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    'mean': valid_data.mean(),
                    'std': valid_data.std()
                }
            else:
                stats = {'min': None, 'max': None, 'mean': None, 'std': None}
            band_stats.append(stats)
        
        band_info['stats'] = band_stats
        
        return band_info


def guess_band_indices(file_or_files):
    """
    Guess band indices (red, green, blue, NIR) based on common naming conventions
    
    Parameters:
    file_or_files: Either a single multiband file path or a list of single-band file paths
    
    Returns:
    dict: Dictionary with guessed band indices
    """
    guess = {
        'red': None,
        'green': None,
        'blue': None,
        'nir': None
    }
    
    if isinstance(file_or_files, str):
        # Multiband file case
        with rasterio.open(file_or_files) as src:
            # Try to guess based on metadata or common conventions
            # For Sentinel-2: B04=Red, B03=Green, B02=Blue, B08=NIR
            # For Landsat 8: B4=Red, B3=Green, B2=Blue, B5=NIR
            
            # Default assumption for 3+ band rasters
            if src.count >= 4:
                # Most common order: RGB-NIR
                guess['red'] = 1
                guess['green'] = 2
                guess['blue'] = 3
                guess['nir'] = 4
            elif src.count >= 3:
                # RGB only
                guess['red'] = 1
                guess['green'] = 2
                guess['blue'] = 3
    else:
        # List of files case
        for i, file_path in enumerate(file_or_files):
            filename = os.path.basename(file_path).lower()
            
            # Check for band indicators in filename
            if 'red' in filename or '_b04_' in filename or '_b4_' in filename or '_b4.' in filename:
                guess['red'] = i
            elif 'green' in filename or '_b03_' in filename or '_b3_' in filename or '_b3.' in filename:
                guess['green'] = i
            elif 'blue' in filename or '_b02_' in filename or '_b2_' in filename or '_b2.' in filename:
                guess['blue'] = i
            elif 'nir' in filename or '_b08_' in filename or '_b8_' in filename or '_b8.' in filename or '_b5_' in filename:
                guess['nir'] = i
    
    return guess