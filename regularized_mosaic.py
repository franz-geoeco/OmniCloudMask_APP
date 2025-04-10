import pathlib
from typing import Union, Optional, Callable, List, Dict, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
import os
import s2mosaic
import streamlit as st
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import pandas as pd

def create_regularized_mosaics(
    grid_id: str,
    start_date: datetime,
    end_date: datetime,
    interval_type: str = 'month',
    interval_value: int = 1,
    interval_duration_days: int = 30,
    output_dir: Union[str, pathlib.Path, None] = None,
    sort_method: str = 'valid_data',
    sort_function: Optional[Callable] = None,
    mosaic_method: str = 'mean',
    required_bands: List[str] = ['B04', 'B03', 'B02', 'B08'],
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 1,
    ocm_inference_dtype: str = 'bf16',
    debug_cache: bool = False,
    additional_query: Dict[str, Any] = {'eo:cloud_cover': {'lt': 100}},
    stack_output: bool = True,
    stack_filename: Optional[str] = None,
    skip_existing: bool = False,
    mask_geometry: Optional[Union[Dict, List[Dict]]] = None,
    mask_path: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> List[Union[Tuple[np.ndarray, Dict[str, Any]], pathlib.Path]]:
    """
    Create a stack of Sentinel-2 mosaics at regular intervals with optional masking.
    
    Args:
        grid_id (str): The ID of the grid area for which to create the mosaic (e.g., "50HMH").
        start_date (datetime): The start date for mosaic generation.
        end_date (datetime): The end date for mosaic generation.
        interval_type (str, optional): Type of interval between mosaics. Options are "day", "week", "month", "year".
            Defaults to "month".
        interval_value (int, optional): Value of interval (e.g., every 2 months). Defaults to 1.
        interval_duration_days (int, optional): Duration in days for each interval's data collection. Defaults to 30.
        output_dir (Optional[Union[Path, str]], optional): Directory to save output GeoTIFFs.
            If None, mosaics are not saved to disk. Defaults to None.
        sort_method (str, optional): Method to sort scenes. Options are "valid_data", "oldest", or "newest".
            Defaults to "valid_data".
        sort_function (Callable, optional): Custom sorting function. If provided, overrides sort_method.
        mosaic_method (str, optional): Method to create the mosaic. Options are "mean" or "first".
            Defaults to "mean".
        required_bands (List[str], optional): List of required spectral bands.
            Defaults to ["B04", "B03", "B02", "B08"] (Red, Green, Blue, NIR).
        no_data_threshold (float, optional): Threshold for no data values. Defaults to 0.01.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
        ocm_batch_size (int, optional): Batch size for OCM inference. Defaults to 1.
        ocm_inference_dtype (str, optional): Data type for OCM inference. Defaults to "bf16".
        debug_cache (bool, optional): Whether to cache downloads for faster debugging. Defaults to False.
        additional_query (Dict[str, Any], optional): Additional query parameters for STAC API.
            Defaults to {"eo:cloud_cover": {"lt": 100}}.
        stack_output (bool, optional): Whether to stack all mosaics into a single NetCDF or multiband GeoTIFF.
            Defaults to True.
        stack_filename (Optional[str], optional): Filename for the stacked output if stack_output is True.
            If None, a default name will be generated. Defaults to None.
        skip_existing (bool, optional): Whether to skip generating mosaics if they already exist.
            Defaults to False.
        mask_geometry (Optional[Union[Dict, List[Dict]]], optional): GeoJSON-like geometry to mask the output.
            Defaults to None.
        mask_path (Optional[str], optional): Path to a vector file to use for masking. Defaults to None.
        bbox (Optional[Tuple[float, float, float, float]], optional): Bounding box coordinates (minx, miny, maxx, maxy)
            to mask the output. Defaults to None.
    
    Returns:
        List[Union[Tuple[np.ndarray, Dict[str, Any]], pathlib.Path]]: List of generated mosaics or paths to saved files.
    
    Raises:
        ValueError: If interval_type is not one of "day", "week", "month", "year".
        Exception: If no scenes are found for a specific time range.
    """
    
    # Validate interval_type
    valid_interval_types = ["day", "week", "month", "year"]
    if interval_type not in valid_interval_types:
        raise ValueError(f"interval_type must be one of {valid_interval_types}")
    
    # Create output directory if specified
    if output_dir is not None:
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize list to store results
    results = []
    
    # Process mask_geometry from various inputs
    geometry_to_use = None
    if mask_geometry:
        geometry_to_use = mask_geometry
    elif mask_path:
        try:
            mask_gdf = gpd.read_file(mask_path)
            geometry_to_use = mask_gdf.geometry.__geo_interface__
        except Exception as e:
            st.warning(f"Error loading mask file: {str(e)}. Continuing without masking.")
    elif bbox:
        # Convert bbox to geometry
        minx, miny, maxx, maxy = bbox
        bbox_geom = box(minx, miny, maxx, maxy)
        geometry_to_use = [bbox_geom.__geo_interface__]
    
    # Create a list of start dates for each interval
    current_date = start_date
    interval_start_dates = []
    
    while current_date < end_date:
        interval_start_dates.append(current_date)
        
        if interval_type == "day":
            current_date += timedelta(days=interval_value)
        elif interval_type == "week":
            current_date += timedelta(weeks=interval_value)
        elif interval_type == "month":
            # Handle month addition (awkward in Python)
            year = current_date.year + ((current_date.month - 1 + interval_value) // 12)
            month = (current_date.month - 1 + interval_value) % 12 + 1
            current_date = datetime(year, month, current_date.day)
        elif interval_type == "year":
            current_date = datetime(current_date.year + interval_value, current_date.month, current_date.day)
    
    # Generate mosaics for each interval
    for i, interval_start in enumerate(interval_start_dates):
        st.text(f"Processing interval {i+1}/{len(interval_start_dates)}: {interval_start.strftime('%Y-%m-%d')}")
        
        # Set output filename if saving to disk
        interval_filename = None
        if output_dir is not None:
            interval_filename = output_dir / f"{grid_id}_{interval_start.strftime('%Y%m%d')}.tif"
            
            # Skip if file exists and skip_existing is True
            if skip_existing and interval_filename.exists():
                st.text(f"  Skipping existing file: {interval_filename}")
                results.append(interval_filename)
                continue
        
        try:
            # Call the mosaic function
            result = s2mosaic.mosaic(
                grid_id=grid_id,
                start_year=interval_start.year,
                start_month=interval_start.month,
                start_day=interval_start.day,
                output_dir=output_dir,
                sort_method=sort_method,
                sort_function=sort_function,
                mosaic_method=mosaic_method,
                duration_days=interval_duration_days,
                required_bands=required_bands,
                no_data_threshold=no_data_threshold,
                overwrite=overwrite,
                ocm_batch_size=ocm_batch_size,
                ocm_inference_dtype=ocm_inference_dtype,
                debug_cache=debug_cache,
                additional_query=additional_query
            )
            
            # Apply masking if geometry is provided and we have a file output
            if geometry_to_use and output_dir is not None and isinstance(result, pathlib.Path):
                try:
                    # Open the generated file
                    with rasterio.open(result) as src:
                        # Check CRS of mask and raster, reproject if needed
                        mask_data, mask_transform = mask(src, geometry_to_use, crop=True)
                        
                        # Prepare output metadata for masked file
                        masked_meta = src.meta.copy()
                        masked_meta.update({
                            "height": mask_data.shape[1],
                            "width": mask_data.shape[2],
                            "transform": mask_transform
                        })
                        
                        # Create new filename for masked output
                        masked_filename = output_dir / f"{grid_id}_{interval_start.strftime('%Y%m%d')}_masked.tif"
                        
                        # Write masked data
                        with rasterio.open(masked_filename, "w", **masked_meta) as dest:
                            dest.write(mask_data)
                        
                        # Replace the result with the masked file path
                        results.append(masked_filename)
                        st.text(f"  Successfully created masked mosaic for {interval_start.strftime('%Y-%m-%d')}")
                        
                        # Optionally: remove unmasked file to save space
                        if os.path.exists(result):
                            os.remove(result)
                    
                except Exception as e:
                    st.warning(f"  Error applying mask: {str(e)}. Using unmasked output.")
                    results.append(result)
            else:
                results.append(result)
                st.text(f"  Successfully created mosaic for {interval_start.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            st.text(f"  Failed to create mosaic for {interval_start.strftime('%Y-%m-%d')}: {str(e)}")
            continue
    
    # Create a stacked output if requested
    if stack_output and output_dir is not None and len(results) > 0:
        try:
            import rasterio
            from rasterio.merge import merge
            import xarray as xr
            import pandas as pd
            
            # Default stack filename
            if stack_filename is None:
                stack_filename = f"{grid_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_stack"
            
            # Determine if we should create NetCDF or multiband GeoTIFF
            if required_bands and len(required_bands) > 1:
                # Create a NetCDF with proper dimensions for multispectral data
                st.text("Creating NetCDF stack with time and band dimensions...")
                
                # Open the first file to get metadata
                datasets = []
                times = []
                for j, result_path in enumerate(results):
                    if isinstance(result_path, pathlib.Path) and os.path.exists(result_path):
                        with rasterio.open(result_path) as src:
                            datasets.append(src.read())
                            # Use the actual interval date for the time dimension
                            if j < len(interval_start_dates):
                                times.append(pd.to_datetime(interval_start_dates[j]))
                            else:
                                # Fallback if we somehow have more results than intervals
                                times.append(pd.to_datetime(f"{j+1}", unit='D', origin=pd.Timestamp(start_date)))
                
                if datasets:
                    # Create xarray dataset
                    stack_data = np.stack(datasets, axis=0)  # (time, band, height, width)
                    
                    # Create xarray DataArray
                    da = xr.DataArray(
                        stack_data,
                        dims=('time', 'band', 'y', 'x'),
                        coords={
                            'time': times,
                            'band': required_bands if len(required_bands) == stack_data.shape[1] else range(stack_data.shape[1])
                        }
                    )
                    
                    # Save as NetCDF
                    netcdf_path = output_dir / f"{stack_filename}.nc"
                    da.to_netcdf(netcdf_path)
                    st.text(f"Saved NetCDF stack to {netcdf_path}")
                    
                    # Add the stack file to results
                    results.append(netcdf_path)
                
            else:
                # Create a multiband GeoTIFF with time as bands
                st.text("Creating multiband GeoTIFF stack with time as bands...")
                
                # Open all the files with rasterio
                src_files_to_mosaic = []
                for result_path in results:
                    if isinstance(result_path, pathlib.Path) and os.path.exists(result_path):
                        src = rasterio.open(result_path)
                        src_files_to_mosaic.append(src)
                
                if src_files_to_mosaic:
                    # Merge into a multiband raster
                    mosaic, out_trans = merge(src_files_to_mosaic)
                    
                    # Copy the metadata
                    out_meta = src_files_to_mosaic[0].meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        "count": len(src_files_to_mosaic)
                    })
                    
                    # Write the stacked tiff
                    stack_path = output_dir / f"{stack_filename}.tif"
                    with rasterio.open(stack_path, "w", **out_meta) as dest:
                        for i in range(mosaic.shape[0]):
                            dest.write(mosaic[i], i + 1)
                    
                    # Close all the open files
                    for src in src_files_to_mosaic:
                        src.close()
                    
                    st.text(f"Saved multiband GeoTIFF stack to {stack_path}")
                    
                    # Add the stack file to results
                    results.append(stack_path)
        
        except ImportError:
            st.warning("Warning: rasterio, pandas, or xarray not installed. Stack creation skipped.")
        except Exception as e:
            st.warning(f"Error creating stack: {str(e)}")
    
    return results