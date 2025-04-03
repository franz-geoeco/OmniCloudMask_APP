import streamlit as st
import os
import glob
import re
from datetime import datetime
import time
import shutil
from collections import defaultdict

# Import custom modules
from file_utils import find_raster_files, is_multiband_file, group_files_by_date_tile
from processing import process_multiband_file, process_single_band_files

def main():
    st.set_page_config(layout="wide", page_title="Cloud Masking App")
    
    st.title("Satellite Image Cloud Masking Tool")
    
    # Add information about OmniCloudMask
    with st.expander("About OmniCloudMask", expanded=True):
        st.markdown("""
        ### OmniCloudMask
        
        This tool uses **OmniCloudMask** (OCM), a sensor-agnostic deep learning model that segments clouds and cloud shadows.
        
        OmniCloudMask demonstrates robust state-of-the-art performance across various satellite platforms when classifying clear, cloud, and shadow classes, with balanced overall accuracy values across:
        - **Landsat**: 91.5% clear, 91.5% cloud, and 75.2% shadow
        - **Sentinel-2**: 92.2% clear, 91.2% cloud, and 80.5% shadow
        - **PlanetScope**: 96.9% clear, 98.8% cloud, and 97.4% shadow
        
        **GitHub**: [https://github.com/DPIRD-DMA/OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
        
        **Paper**: [Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask](https://www.sciencedirect.com/science/article/pii/S0034425725000987)
        
        **Citation**:
        ```
        @article{WRIGHT2025114694,
        title = {Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask},
        journal = {Remote Sensing of Environment},
        volume = {322},
        pages = {114694},
        year = {2025},
        issn = {0034-4257},
        doi = {https://doi.org/10.1016/j.rse.2025.114694}, 
        url = {https://www.sciencedirect.com/science/article/pii/S0034425725000987},
        author = {Nicholas Wright and John M.A. Duncan and J. Nik Callow and Sally E. Thompson and Richard J. George},
        keywords = {Sensor-agnostic, Deep learning, Cloud, Shadow, Sentinel-2, Landsat, PlanetScope}
        }
        ```
        """)
    
    st.markdown("""
    This application helps you mask clouds in satellite imagery. 
    You can process multiband GeoTIFFs or folders with multiple 
    single-band images. The app identifies cloud pixels and sets 
    them to NoData in the output images.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Processing Options")
        
        processing_mode = st.radio(
            "Select Processing Mode",
            ["Single File", "Folder Processing"]
        )
        
        # Add resampling factor option
        resampling_factor = st.select_slider(
            "Resampling Factor (for processing speed)",
            options=[1, 2, 4, 8, 16],
            value=2,
            help="Higher values process faster but with less precision. 1 = no resampling"
        )
        
        # Check if GPU is available
        import torch
        has_cuda = torch.cuda.is_available()
        device = "cuda" if has_cuda else "cpu"
        
        st.write(f"Processing device: {'GPU (CUDA)' if has_cuda else 'CPU (GPU not available)'}")
        
        # Advanced cloud detection options
        st.subheader("Cloud Detection Options")
        
        with st.expander("Advanced Settings"):
            patch_size = st.number_input("Patch Size", min_value=100, max_value=2000, value=1000, 
                                        help="Size of the patches for inference")
            
            patch_overlap = st.number_input("Patch Overlap", min_value=0, max_value=1000, value=300,
                                           help="Overlap between patches for inference")
            
            inference_batch_size = st.number_input("Inference Batch Size", min_value=1, max_value=16, value=1,
                                                 help="Number of patches to process in a batch")
            
            inference_dtype = st.selectbox("Inference Data Type", 
                                          options=["float32", "float16", "bfloat16"],
                                          index=0,
                                          help="Data type for inference")
            
            export_confidence = st.checkbox("Export Confidence Maps", value=False,
                                          help="If checked, exports confidence maps instead of predicted classes")
            
            softmax_output = st.checkbox("Apply Softmax", value=True,
                                        help="If checked, applies a softmax to the output (only used with confidence maps)")
            
            no_data_value = st.number_input("No Data Value", value=0,
                                           help="Value within input scenes that specifies no data region")
            
            apply_no_data_mask = st.checkbox("Apply No Data Mask", value=True,
                                           help="If checked, applies a no-data mask to the predictions")
            
            model_source = st.selectbox("Model Download Source",
                                       options=["google_drive", "hugging_face"],
                                       index=0,
                                       help="Source from which to download model weights")
        
        # Create a dictionary of detection options to pass to processing functions
        detection_options = {
            "patch_size": patch_size,
            "patch_overlap": patch_overlap,
            "batch_size": inference_batch_size,
            "inference_device": device,
            "mosaic_device": None,  # Will use inference device
            "inference_dtype": inference_dtype,
            "export_confidence": export_confidence,
            "softmax_output": softmax_output,
            "no_data_value": no_data_value,
            "apply_no_data_mask": apply_no_data_mask,
            "model_download_source": model_source
        }
        
        if processing_mode == "Single File":
            st.subheader("Single File Input")
            uploaded_file = st.file_uploader("Upload a multiband GeoTIFF file", type=["tif", "tiff"])
            
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                temp_dir = "temp_upload"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Get band count
                import rasterio
                with rasterio.open(temp_path) as src:
                    band_count = src.count
                
                # Band selection
                st.subheader("Band Selection")
                red_band_idx = st.selectbox("Red Band Index (1-based)", options=range(1, band_count+1), index=2 if band_count >= 3 else 0)
                green_band_idx = st.selectbox("Green Band Index (1-based)", options=range(1, band_count+1), index=1 if band_count >= 3 else 0)
                nir_band_idx = st.selectbox("NIR Band Index (1-based)", options=range(1, band_count+1), index=3 if band_count >= 4 else 0)
        
        else:  # Folder Processing
            st.subheader("Folder Input")
            folder_path = st.text_input("Enter folder path containing satellite imagery")
            
            if folder_path and os.path.isdir(folder_path):
                # Find all raster files
                all_files = find_raster_files(folder_path)
                st.write(f"Found {len(all_files)} raster files")
                
                # Check if there are multiband files
                multiband_files = [f for f in all_files if is_multiband_file(f)]
                
                if multiband_files:
                    st.write(f"Found {len(multiband_files)} multiband files")
                    
                    # Allow user to select a multiband file
                    selected_file = st.selectbox(
                        "Select a multiband file to process", 
                        multiband_files,
                        format_func=os.path.basename
                    )
                    
                    # Get band count
                    import rasterio
                    with rasterio.open(selected_file) as src:
                        band_count = src.count
                    
                    # Band selection
                    st.subheader("Band Selection")
                    red_band_idx = st.selectbox("Red Band Index (1-based)", options=range(1, band_count+1), index=2 if band_count >= 3 else 0)
                    green_band_idx = st.selectbox("Green Band Index (1-based)", options=range(1, band_count+1), index=1 if band_count >= 3 else 0)
                    nir_band_idx = st.selectbox("NIR Band Index (1-based)", options=range(1, band_count+1), index=3 if band_count >= 4 else 0)
                    
                    # Process all multiband files or just selected
                    process_all = st.checkbox("Process all multiband files in folder", value=False)
                    
                else:
                    # Try to group single-band files
                    grouped_files = group_files_by_date_tile(all_files)
                    
                    if grouped_files:
                        st.write(f"Grouped files into {len(grouped_files)} time periods")
                        
                        # Let user select a group to view
                        selected_group = st.selectbox(
                            "Select a time period to view files",
                            list(grouped_files.keys())
                        )
                        
                        if selected_group:
                            st.write(f"Files in group: {', '.join([os.path.basename(f) for f in grouped_files[selected_group]])}")
                            
                            # Let user select red, green, and NIR bands from the files
                            file_options = [os.path.basename(f) for f in grouped_files[selected_group]]
                            
                            st.subheader("Band Selection")
                            red_file_idx = st.selectbox("Select Red Band File", options=range(len(file_options)), format_func=lambda x: file_options[x])
                            green_file_idx = st.selectbox("Select Green Band File", options=range(len(file_options)), format_func=lambda x: file_options[x])
                            nir_file_idx = st.selectbox("Select NIR Band File", options=range(len(file_options)), format_func=lambda x: file_options[x])
                            
                            # Process all groups or just selected
                            process_all = st.checkbox("Process all time periods in folder", value=False)
        
        # Output directory
        output_dir = st.text_input("Output Directory", "masked_output")
        
        # Start processing button
        process_button = st.button("Start Processing")
    
    # Main content area
    if process_button:
        if processing_mode == "Single File" and uploaded_file is not None:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            with st.spinner("Processing file..."):
                success = process_multiband_file(temp_path, red_band_idx, green_band_idx, nir_band_idx, output_dir)
                
                if success:
                    st.success(f"Processing complete! Results saved to {output_dir}")
                else:
                    st.error("Processing failed.")
        
        elif processing_mode == "Folder Processing" and folder_path and os.path.isdir(folder_path):
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            if multiband_files:
                if process_all:
                    # Process all multiband files
                    with st.spinner(f"Processing {len(multiband_files)} multiband files..."):
                        progress_bar = st.progress(0)
                        
                        for i, file_path in enumerate(multiband_files):
                            st.write(f"Processing {os.path.basename(file_path)}...")
                            success = process_multiband_file(file_path, red_band_idx, green_band_idx, nir_band_idx, output_dir, resampling_factor, device, detection_options)
                            progress_bar.progress((i + 1) / len(multiband_files))
                        
                        st.success(f"Processing complete! Results saved to {output_dir}")
                
                else:
                    # Process just the selected multiband file
                    with st.spinner("Processing file..."):
                        success = process_multiband_file(selected_file, red_band_idx, green_band_idx, nir_band_idx, output_dir, resampling_factor, device, detection_options)
                        
                        if success:
                            st.success(f"Processing complete! Results saved to {output_dir}")
                        else:
                            st.error("Processing failed.")
            
            else:
                # Process single-band files
                if process_all:
                    # Process all groups
                    with st.spinner(f"Processing {len(grouped_files)} time periods..."):
                        progress_bar = st.progress(0)
                        
                        for i, (group_name, files) in enumerate(grouped_files.items()):
                            st.write(f"Processing time period: {group_name}...")
                            
                            # Need to ensure we have the right files for each group
                            red_band_file = files[0]  # Default to first file
                            green_band_file = files[0]  # Default to first file
                            nir_band_file = files[0]  # Default to first file
                            
                            # Try to infer based on filenames
                            for file in files:
                                if "B04" in os.path.basename(file):
                                    red_band_file = file
                                elif "B03" in os.path.basename(file):
                                    green_band_file = file
                                elif "B08" in os.path.basename(file):
                                    nir_band_file = file
                            
                            # Create a dictionary of all files for this group
                            files_dict = {os.path.basename(f): f for f in files}
                            
                            process_single_band_files(files_dict, red_band_file, green_band_file, nir_band_file, output_dir, resampling_factor, device, detection_options)
                            progress_bar.progress((i + 1) / len(grouped_files))
                        
                        st.success(f"Processing complete! Results saved to {output_dir}")
                
                else:
                    # Process just the selected group
                    with st.spinner("Processing selected time period..."):
                        files = grouped_files[selected_group]
                        
                        # Get the selected files
                        red_band_file = files[red_file_idx]
                        green_band_file = files[green_file_idx]
                        nir_band_file = files[nir_file_idx]
                        
                        # Create a dictionary of all files for this group
                        files_dict = {os.path.basename(f): f for f in files}
                        
                        success = process_single_band_files(files_dict, red_band_file, green_band_file, nir_band_file, output_dir, resampling_factor, device, detection_options)
                        
                        if success:
                            st.success(f"Processing complete! Results saved to {output_dir}")
                        else:
                            st.error("Processing failed.")


if __name__ == "__main__":
    main()