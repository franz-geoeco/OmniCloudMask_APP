import os
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import streamlit as st
import torch

# Import custom modules
# Import custom modules
from omnicloudmask import predict_from_array
from visualization import plot_results, display_cloud_statistics

def process_multiband_file(file_path, red_band_idx, green_band_idx, nir_band_idx, output_dir, resampling_factor=2, device="cuda", detection_options=None):
    """
    Process a multiband file for cloud masking
    
    Parameters:
    file_path (str): Path to the multiband file
    red_band_idx (int): Index of the red band (1-based)
    green_band_idx (int): Index of the green band (1-based)
    nir_band_idx (int): Index of the NIR band (1-based)
    output_dir (str): Directory to save output files
    resampling_factor (int): Factor by which to resample the image for processing (default=2)
    device (str): Device to use for inference ("cuda" or "cpu")
    detection_options (dict): Dictionary of options to pass to predict_from_array
    
    Returns:
    bool: True if processing was successful, False otherwise
    """
    st.write(f"Processing multiband file: {os.path.basename(file_path)}")
    
    # Extract mask class options
    mask_classes = detection_options.pop("mask_classes", {"thick_cloud": True, "thin_cloud": True, "cloud_shadow": True})
    # Extract compression settings if provided
    compression_settings = detection_options.pop("compression", {"type": "lzw", "level": None})

    try:
        with rasterio.open(file_path) as src:
            # Read the specified bands
            red_band = src.read(red_band_idx)
            green_band = src.read(green_band_idx)
            nir_band = src.read(nir_band_idx)
            
            # Get metadata
            metadata = src.meta
            original_transform = src.transform
            
            # IMPORTANT: Extract compression settings from source file
            if hasattr(src, 'compression'):
                original_compression = src.compression.value if src.compression else 'lzw'
            else:
                # Use provided compression settings or default to lzw
                original_compression = compression_settings["type"] if compression_settings["type"] != "none" else 'lzw'
            
            # Set compression level if using a compression type that supports levels
            compression_level = None
            if original_compression in ['deflate', 'zstd'] and compression_settings.get("level"):
                compression_level = compression_settings["level"]
            
            # Calculate new dimensions based on resampling factor
            if resampling_factor > 1:
                new_height = src.height // resampling_factor
                new_width = src.width // resampling_factor
                st.info(f"Resampling image from {src.height}x{src.width} to {new_height}x{new_width} for processing")
            else:
                # No resampling
                new_height = src.height
                new_width = src.width
                st.info("Processing at full resolution (no resampling)")
            
            # Create updated metadata for resampled data
            resampled_metadata = src.meta.copy()
            resampled_metadata.update({
                'height': new_height,
                'width': new_width,
                'transform': rasterio.transform.from_bounds(
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                    new_width, new_height
                )
            })
            
            # Resample bands to lower resolution for cloud detection
            resampled_bands = []
            for band_data in [red_band, green_band, nir_band]:
                # Reshape to add band dimension if not present
                if len(band_data.shape) == 2:
                    band_data = np.expand_dims(band_data, 0)
                    
                # Resample
                resampled_data = np.zeros((1, new_height, new_width), dtype=band_data.dtype)
                reproject(
                    source=band_data,
                    destination=resampled_data,
                    src_transform=original_transform,
                    dst_transform=resampled_metadata['transform'],
                    src_crs=src.crs,
                    dst_crs=src.crs,
                    resampling=Resampling.average
                )
                resampled_bands.append(resampled_data[0])
            
            # Stack bands for cloud detection
            resampled_stack = np.stack(resampled_bands, axis=0).astype(np.float32)
            
            # Trim no-data values
            no_data_value = src.nodata if src.nodata is not None else -9999
            any_nodata = np.any(resampled_stack == no_data_value, axis=0)
            valid_mask = ~any_nodata
            
            # Find the bounding box of valid data
            rows, cols = np.where(valid_mask)
            if len(rows) > 0 and len(cols) > 0:
                row_min, row_max = rows.min(), rows.max() + 1
                col_min, col_max = cols.min(), cols.max() + 1
                
                # Trim the data to the valid region
                trimmed_stack = resampled_stack[:, row_min:row_max, col_min:col_max]
                
                # Update transform for trimmed data
                old_transform = resampled_metadata['transform']
                new_transform = rasterio.transform.from_origin(
                    old_transform.c + col_min * old_transform.a,
                    old_transform.f + row_min * old_transform.e,
                    old_transform.a,
                    old_transform.e
                )
                
                # Update metadata for trimmed data
                trimmed_metadata = resampled_metadata.copy()
                trimmed_metadata.update({
                    'height': row_max - row_min,
                    'width': col_max - col_min,
                    'transform': new_transform,
                    'count': resampled_stack.shape[0]
                })
                
                # Prepare detection options
                if detection_options is None:
                    detection_options = {}
                
                # Extract application-specific options before passing to predict_from_array
                save_cloud_mask = detection_options.pop("save_cloud_mask", True) if detection_options else True
                
                # Create a copy of detection options to avoid modifying the original
                predict_options = detection_options.copy() if detection_options else {}
                
                # Remove export_confidence and softmax_output if they exist
                if "export_confidence" in predict_options:
                    predict_options.pop("export_confidence")
                if "softmax_output" in predict_options:
                    predict_options.pop("softmax_output")
                
                # Force these values to ensure we always get a binary mask
                predict_options["export_confidence"] = False
                predict_options["softmax_output"] = False
                
                # Handle inference_dtype conversion from string to torch.dtype
                if "inference_dtype" in predict_options and isinstance(predict_options["inference_dtype"], str):
                    dtype_str = predict_options["inference_dtype"]
                    if dtype_str == "float32":
                        predict_options["inference_dtype"] = torch.float32
                    elif dtype_str == "float16":
                        predict_options["inference_dtype"] = torch.float16
                    elif dtype_str == "bfloat16":
                        if hasattr(torch, "bfloat16"):
                            predict_options["inference_dtype"] = torch.bfloat16
                        else:
                            st.warning("bfloat16 not supported in your PyTorch version, falling back to float16")
                            predict_options["inference_dtype"] = torch.float16
                
                # Set inference device
                predict_options["inference_device"] = device
                
                # Call predict_from_array with valid options only
                pred_mask_res = predict_from_array(
                    trimmed_stack,
                    **predict_options
                )
                
                # Create a visualization
                rgb_for_plot = np.stack([trimmed_stack[2], trimmed_stack[0], trimmed_stack[1]], axis=0)  # NIR, Red, Green
                fig = plot_results(rgb_for_plot, pred_mask_res[0])
                st.pyplot(fig)

                # Display cloud statistics
                display_cloud_statistics(pred_mask_res[0])

                # Reconstruct full resolution mask
                full_mask = np.full((resampled_metadata['height'], resampled_metadata['width']), -9999, dtype=np.float32)
                full_mask[row_min:row_max, col_min:col_max] = pred_mask_res[0]
                
                # Upsample mask to original resolution
                upsampled_mask = np.zeros((src.height, src.width), dtype=np.float32)
                reproject(
                    source=full_mask,
                    destination=upsampled_mask,
                    src_transform=resampled_metadata['transform'],
                    dst_transform=original_transform,
                    src_crs=src.crs,
                    dst_crs=src.crs,
                    src_nodata=-9999,
                    dst_nodata=-9999,
                    resampling=Resampling.nearest
                )
                
                # Apply mask to all bands in the original file
                masked_data = np.zeros((src.count, src.height, src.width), dtype=src.dtypes[0])
                for i in range(1, src.count + 1):
                    band_data = src.read(i)
                    
                    # Create validity mask
                    validity_mask = (band_data != no_data_value) & (upsampled_mask != -9999)
                    
                    # Apply cloud mask: set cloud pixels to NoData
                    masked_band = band_data.copy()
                    # Create cloud pixels mask based on selected classes
                    cloud_pixels_mask = np.zeros_like(validity_mask, dtype=bool)
                    
                    # Class 1: Thick Cloud
                    if mask_classes.get("thick_cloud", True):
                        cloud_pixels_mask |= (upsampled_mask == 1) & validity_mask
                    
                    # Class 2: Thin Cloud
                    if mask_classes.get("thin_cloud", True):
                        cloud_pixels_mask |= (upsampled_mask == 2) & validity_mask
                    
                    # Class 3: Cloud Shadow
                    if mask_classes.get("cloud_shadow", True):
                        cloud_pixels_mask |= (upsampled_mask == 3) & validity_mask
                    
                    # Apply the mask
                    masked_band[cloud_pixels_mask] = no_data_value
                    
                    masked_data[i-1] = masked_band
                
                # Prepare output metadata
                output_metadata = src.meta.copy()
                
                # IMPORTANT: Apply compression to output files
                if original_compression:
                    output_metadata.update({'compress': original_compression})
                    
                    # Add compression level if applicable
                    if compression_level is not None and original_compression in ['deflate', 'zstd']:
                        output_metadata.update({'compress_level': compression_level})
                
                # Save masked file
                os.makedirs(output_dir, exist_ok=True)
                output_filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, output_filename)
                
                with rasterio.open(output_path, 'w', **output_metadata) as dst:
                    dst.write(masked_data)
                
                # Save cloud mask as separate file if option is enabled
                if save_cloud_mask:
                    mask_filename = os.path.splitext(output_filename)[0] + '_CLOUDMASK.tif'
                    mask_path = os.path.join(output_dir, mask_filename)
                    
                    mask_metadata = src.meta.copy()
                    mask_metadata.update({
                        'count': 1,
                        'dtype': 'uint8',
                        'nodata': 255
                    })
                    
                    # Apply compression to mask file too
                    if original_compression:
                        mask_metadata.update({'compress': original_compression})
                        
                        # Add compression level if applicable
                        if compression_level is not None and original_compression in ['deflate', 'zstd']:
                            mask_metadata.update({'compress_level': compression_level})
                    
                    binary_mask = np.where(upsampled_mask == -9999, 255, upsampled_mask.astype(np.uint8))
                    
                    with rasterio.open(mask_path, 'w', **mask_metadata) as dst:
                        dst.write(binary_mask, 1)
                    
                    st.success(f"Processed file saved to: {output_path}")
                    st.success(f"Cloud mask saved to: {mask_path}")
                else:
                    st.success(f"Processed file saved to: {output_path}")
                
                return True
            else:
                st.error("No valid data found in the raster.")
                return False
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False


def process_single_band_files(files_dict, red_band_file, green_band_file, nir_band_file, output_dir, resampling_factor=2, device="cuda", detection_options=None):
    """
    Process a set of single band files for cloud masking
    
    Parameters:
    files_dict (dict): Dictionary of file paths keyed by filename
    red_band_file (str): Path to the red band file
    green_band_file (str): Path to the green band file
    nir_band_file (str): Path to the NIR band file
    output_dir (str): Directory to save output files
    resampling_factor (int): Factor by which to resample the image for processing (default=2)
    device (str): Device to use for inference ("cuda" or "cpu")
    detection_options (dict): Dictionary of options to pass to predict_from_array
    
    Returns:
    bool: True if processing was successful, False otherwise
    """
    st.write(f"Processing single-band files")
    
    # Extract mask class options
    mask_classes = detection_options.pop("mask_classes", {"thick_cloud": True, "thin_cloud": True, "cloud_shadow": True})
    # Extract compression settings if provided
    compression_settings = detection_options.pop("compression", {"type": "lzw", "level": None})

    try:
        # Get metadata from red band file
        with rasterio.open(red_band_file) as red_src:
            red_band = red_src.read(1)
            metadata = red_src.meta
            original_transform = red_src.transform
            
            # IMPORTANT: Extract compression settings from source file
            if hasattr(red_src, 'compression'):
                original_compression = red_src.compression.value if red_src.compression else 'lzw'
            else:
                # Use provided compression settings or default to lzw
                original_compression = compression_settings["type"] if compression_settings["type"] != "none" else 'lzw'
            
            # Set compression level if using a compression type that supports levels
            compression_level = None
            if original_compression in ['deflate', 'zstd'] and compression_settings.get("level"):
                compression_level = compression_settings["level"]
                
            no_data_value = red_src.nodata if red_src.nodata is not None else -9999
        
        with rasterio.open(green_band_file) as green_src:
            green_band = green_src.read(1)
        
        with rasterio.open(nir_band_file) as nir_src:
            nir_band = nir_src.read(1)
        
        # Calculate new dimensions based on resampling factor
        with rasterio.open(red_band_file) as src:
            if resampling_factor > 1:
                new_height = src.height // resampling_factor
                new_width = src.width // resampling_factor
                st.info(f"Resampling image from {src.height}x{src.width} to {new_height}x{new_width} for processing")
            else:
                # No resampling
                new_height = src.height
                new_width = src.width
                st.info("Processing at full resolution (no resampling)")
            
            # Create updated metadata
            resampled_metadata = src.meta.copy()
            resampled_metadata.update({
                'height': new_height,
                'width': new_width,
                'transform': rasterio.transform.from_bounds(
                    src.bounds.left, src.bounds.bottom,
                    src.bounds.right, src.bounds.top,
                    new_width, new_height
                )
            })
        
        # Prepare a list to collect resampled bands
        resampled_bands = []
        
        # Resample each band to lower resolution for cloud detection
        for band_data in [red_band, green_band, nir_band]:
            # Reshape to add band dimension if not present
            if len(band_data.shape) == 2:
                band_data = np.expand_dims(band_data, 0)
                
            # Resample
            resampled_data = np.zeros((1, new_height, new_width), dtype=band_data.dtype)
            reproject(
                source=band_data,
                destination=resampled_data,
                src_transform=original_transform,
                dst_transform=resampled_metadata['transform'],
                src_crs=resampled_metadata['crs'],
                dst_crs=resampled_metadata['crs'],
                resampling=Resampling.average
            )
            resampled_bands.append(resampled_data[0])
        
        # Stack bands for cloud detection
        resampled_stack = np.stack(resampled_bands, axis=0).astype(np.float32)
        
        # Trim no-data values
        any_nodata = np.any(resampled_stack == no_data_value, axis=0)
        valid_mask = ~any_nodata
        
        # Find the bounding box of valid data
        rows, cols = np.where(valid_mask)
        if len(rows) > 0 and len(cols) > 0:
            row_min, row_max = rows.min(), rows.max() + 1
            col_min, col_max = cols.min(), cols.max() + 1
            
            # Trim the data to the valid region
            trimmed_stack = resampled_stack[:, row_min:row_max, col_min:col_max]
            
            # Update transform for trimmed data
            old_transform = resampled_metadata['transform']
            new_transform = rasterio.transform.from_origin(
                old_transform.c + col_min * old_transform.a,
                old_transform.f + row_min * old_transform.e,
                old_transform.a,
                old_transform.e
            )
            
            # Update metadata for trimmed data
            trimmed_metadata = resampled_metadata.copy()
            trimmed_metadata.update({
                'height': row_max - row_min,
                'width': col_max - col_min,
                'transform': new_transform,
                'count': resampled_stack.shape[0]
            })
            
            # Prepare detection options
            if detection_options is None:
                detection_options = {}
            
            # Extract application-specific options before passing to predict_from_array
            save_cloud_mask = detection_options.pop("save_cloud_mask", True) if detection_options else True
            
            # Create a copy of detection options to avoid modifying the original
            predict_options = detection_options.copy() if detection_options else {}
            
            # Remove export_confidence and softmax_output if they exist
            if "export_confidence" in predict_options:
                predict_options.pop("export_confidence")
            if "softmax_output" in predict_options:
                predict_options.pop("softmax_output")
            
            # Force these values to ensure we always get a binary mask
            predict_options["export_confidence"] = False
            predict_options["softmax_output"] = False
            
            # Handle inference_dtype conversion from string to torch.dtype
            if "inference_dtype" in predict_options and isinstance(predict_options["inference_dtype"], str):
                dtype_str = predict_options["inference_dtype"]
                if dtype_str == "float32":
                    predict_options["inference_dtype"] = torch.float32
                elif dtype_str == "float16":
                    predict_options["inference_dtype"] = torch.float16
                elif dtype_str == "bfloat16":
                    if hasattr(torch, "bfloat16"):
                        predict_options["inference_dtype"] = torch.bfloat16
                    else:
                        st.warning("bfloat16 not supported in your PyTorch version, falling back to float16")
                        predict_options["inference_dtype"] = torch.float16
            
            # Set inference device
            predict_options["inference_device"] = device
            
            # Call predict_from_array with valid options only
            pred_mask_res = predict_from_array(
                trimmed_stack,
                **predict_options
            )
            
            # Create a visualization
            rgb_for_plot = np.stack([trimmed_stack[2], trimmed_stack[0], trimmed_stack[1]], axis=0)  # NIR, Red, Green
            fig = plot_results(rgb_for_plot, pred_mask_res[0])
            st.pyplot(fig)

            # Display cloud statistics
            display_cloud_statistics(pred_mask_res[0])
            
            # Reconstruct full resolution mask
            full_mask = np.full((resampled_metadata['height'], resampled_metadata['width']), -9999, dtype=np.float32)
            full_mask[row_min:row_max, col_min:col_max] = pred_mask_res[0]
            
            # Upsample mask to original resolution
            upsampled_mask = np.zeros((red_band.shape[0], red_band.shape[1]), dtype=np.float32)
            reproject(
                source=full_mask,
                destination=upsampled_mask,
                src_transform=resampled_metadata['transform'],
                dst_transform=original_transform,
                src_crs=resampled_metadata['crs'],
                dst_crs=resampled_metadata['crs'],
                src_nodata=-9999,
                dst_nodata=-9999,
                resampling=Resampling.nearest
            )
            
            # Process all bands in the files dictionary
            for band_key, file_path in files_dict.items():
                with rasterio.open(file_path) as src:
                    band_data = src.read(1)
                    
                    # Create validity mask
                    validity_mask = (band_data != no_data_value) & (upsampled_mask != -9999)
                    
                    # Apply cloud mask: set cloud pixels to NoData
                    masked_band = band_data.copy()
                    
                    # Create cloud pixels mask based on selected classes
                    cloud_pixels_mask = np.zeros_like(validity_mask, dtype=bool)
                    
                    # Class 1: Thick Cloud
                    if mask_classes.get("thick_cloud", True):
                        cloud_pixels_mask |= (upsampled_mask == 1) & validity_mask
                    
                    # Class 2: Thin Cloud
                    if mask_classes.get("thin_cloud", True):
                        cloud_pixels_mask |= (upsampled_mask == 2) & validity_mask
                    
                    # Class 3: Cloud Shadow
                    if mask_classes.get("cloud_shadow", True):
                        cloud_pixels_mask |= (upsampled_mask == 3) & validity_mask
                    
                    # Apply the mask
                    masked_band[cloud_pixels_mask] = no_data_value
                    
                    # Prepare output metadata
                    output_metadata = src.meta.copy()
                    
                    # IMPORTANT: Apply compression to output files
                    if original_compression:
                        output_metadata.update({'compress': original_compression})
                        
                        # Add compression level if applicable
                        if compression_level is not None and original_compression in ['deflate', 'zstd']:
                            output_metadata.update({'compress_level': compression_level})
                    
                    # Save masked file
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = os.path.basename(file_path)
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with rasterio.open(output_path, 'w', **output_metadata) as dst:
                        dst.write(masked_band, 1)
            
            # Save cloud mask as separate file if option is enabled
            if save_cloud_mask:
                base_filename = os.path.basename(red_band_file)
                pattern = r'(.*)_B\d+_(.*)\.tif'
                match = re.search(pattern, base_filename, re.IGNORECASE)
                
                if match:
                    prefix, date_str = match.groups()
                    mask_filename = f"{prefix}_CLOUDMASK_{date_str}.tif"
                else:
                    mask_filename = "cloudmask.tif"
                    
                mask_path = os.path.join(output_dir, mask_filename)
                
                mask_metadata = metadata.copy()
                mask_metadata.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': 255
                })
                
                # IMPORTANT: Apply compression to mask file
                if original_compression:
                    mask_metadata.update({'compress': original_compression})
                    
                    # Add compression level if applicable
                    if compression_level is not None and original_compression in ['deflate', 'zstd']:
                        mask_metadata.update({'compress_level': compression_level})
                
                binary_mask = np.where(upsampled_mask == -9999, 255, upsampled_mask.astype(np.uint8))
                
                with rasterio.open(mask_path, 'w', **mask_metadata) as dst:
                    dst.write(binary_mask, 1)
                
                st.success(f"Processed {len(files_dict)} files. Saved to: {output_dir}")
                st.success(f"Cloud mask saved to: {mask_path}")
            else:
                st.success(f"Processed {len(files_dict)} files. Saved to: {output_dir}")
            
            return True
        else:
            st.error("No valid data found in the raster.")
            return False
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return False