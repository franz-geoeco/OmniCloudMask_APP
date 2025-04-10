# Satellite Image Processing Suite

A Streamlit application for processing satellite imagery with two main components:
1. **Cloud Masking Tool**: Detect and mask clouds using [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
2. **Mosaic Builder**: Create and regularize mosaics from Sentinel-2 imagery with the same advanced cloud masking.

![Application Interface](app_image.png)

## About the Tools

### OmniCloudMask

OmniCloudMask (OCM) is a sensor-agnostic deep learning model that segments clouds and cloud shadows developed by [Nicholas Wright](https://github.com/wrignj08) and [Jordan A. Caraballo-Vega](https://github.com/jordancaraballo). It demonstrates robust state-of-the-art performance across various satellite platforms when classifying clear, cloud, and shadow classes, with balanced overall accuracy values across:
- **Landsat**: 91.5% clear, 91.5% cloud, and 75.2% shadow
- **Sentinel-2**: 92.2% clear, 91.2% cloud, and 80.5% shadow
- **PlanetScope**: 96.9% clear, 98.8% cloud, and 97.4% shadow

**GitHub**: [https://github.com/DPIRD-DMA/OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)

**Paper**: [Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask](https://www.sciencedirect.com/science/article/pii/S0034425725000987)

### S2Mosaic

The Mosaic Builder is based on [S2Mosaic](https://github.com/DPIRD-DMA/S2Mosaic), a Python package for creating Sentinel-2 mosaics with cloud masking using OmniCloudMask. The tool provides:

- Creation of mosaics for specific grid areas and time periods
- Time series generation at regular intervals
- Automatic masking to geographic boundaries
- Merging overlapping areas by averaging pixel values
- Cloud-free composite image creation

## Installation

1. Create a new conda environment:
```bash
conda create -n satprocessing python=3.10
conda activate satprocessing
```

2. Clone this repository:
```bash
git clone https://github.com/geoeco-mlu/OmniCloudMask_APP.git
cd satellite-processing-suite
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit app:
```bash
streamlit run ⌂_Home.py --server.maxUploadSize 1000
```

The app will open in your default web browser, typically at http://localhost:8501.

## Features

### Cloud Masking Tool

- Process single multiband GeoTIFF files or folders of single-band or multi-band files
- Automatic detection of file groups by date/tile
- Customizable resampling for faster processing
- GPU detection and utilization when available
- Advanced cloud detection parameters
- Multi-core processing
- Visualization of results
- Batch processing

### Mosaic Builder

- Create mosaics for specific Sentinel-2 grid tiles and time ranges
- Generate time series mosaics at regular intervals (daily, weekly, monthly, yearly)
- Automatically merge multiple images for the same area with pixel averaging
- Mask outputs to specific geographic boundaries using:
  - Bounding boxes: Simple rectangular areas defined by coordinates
  - Vector files: Complex shapes from uploaded shapefiles or GeoJSON
- Stack time series for analysis
- Export to GeoTIFF or NetCDF formats
- Advanced cloud filtering and mosaic creation options

## Detailed Workflows

### Cloud Masking Workflow

#### 1. Input Selection

The application supports two main processing modes:

#### Single File Mode
- Upload a multiband GeoTIFF file through the interface
- Select the appropriate band indices for Red, Green, and NIR bands
- These bands are used for cloud detection

#### Folder Processing Mode
- Enter a path to a folder containing satellite imagery
- The app automatically detects and groups files by date and tile ID
- For multiband files, select which bands to use
- For single-band files, select which files represent the Red, Green, and NIR bands

#### 2. Processing Options

Various processing options can be configured:

##### Basic Options
- **Resampling Factor**: Controls the resolution at which cloud detection runs
  - Higher values (2, 4, 8, 16) process faster but with less precision
  - Value of 1 means no resampling (full resolution)
- **Output Directory**: Where processed files will be saved (default: "masked_output")

##### Advanced Options
- **Patch Size**: Size of the patches for inference (default: 1000)
- **Patch Overlap**: Overlap between patches for inference (default: 300)
- **Batch Size**: Number of patches to process in a batch (default: 1)
- **Inference Data Type**: Data type for inference (float32, float16, or bfloat16)
- **Export Confidence**: If enabled, exports confidence maps instead of predicted classes
- **Apply Softmax**: Applies softmax to the output when exporting confidence maps
- **No Data Value**: Value that indicates no data in the input images
- **Apply No Data Mask**: Whether to apply a no-data mask to the predictions
- **Model Download Source**: Source for downloading model weights (Google Drive or Hugging Face)

#### 3. Processing Steps

The application performs the following steps for cloud detection and masking:

1. **Data Loading**: Reads the selected bands from the input file(s)
2. **Resampling**: Downsamples the data based on the resampling factor for faster processing
3. **No-Data Trimming**: Identifies and trims regions with no data to improve efficiency
4. **Cloud Detection**: Applies the OmniCloudMask model to detect clouds and cloud shadows
5. **Visualization**: Displays the input image, detected clouds, and overlay
6. **Mask Upsampling**: For resampled data, upsamples the cloud mask back to original resolution
7. **Mask Application**: Sets cloud pixels to NoData value in all bands
8. **Output Generation**:
   - Saves masked versions of all input bands with original filenames
   - Creates a separate cloud mask file with "_CLOUDMASK" suffix

#### 4. Output Files

The application generates the following output files in the specified output directory:

- **Masked Bands**: Original band files with cloud pixels set to NoData
- **Cloud Mask**: A binary GeoTIFF file with classes:
  - 0 = clear
  - 1 = Thick Cloud
  - 2 = Thin Cloud
  - 3 = Cloud Shadow
- **Visualization**: Optional PNG file showing the detection results

### Mosaic Builder Workflow

#### 1. Grid Selection

The application offers three methods to select Sentinel-2 grid tiles:

##### Manual Input
- Directly select grid IDs from the available Sentinel-2 tiles
- Multiple tiles can be selected simultaneously for larger areas

##### Vector File
- Upload a vector file (shapefile, GeoJSON, KML, etc.)
- The app automatically finds all Sentinel-2 grid tiles that intersect with the boundary
- The final output will be masked to exactly match your vector geometry

##### Bounding Box
- Define a geographic bounding box by entering coordinates
- All intersecting grid tiles are identified automatically
- The final output will be clipped to the bounding box boundaries

#### 2. Time Range Selection

Two modes are available:

##### Single Mosaic
- Select a specific start date
- Define a data collection duration (in days)
- Creates a single mosaic for that time period

##### Time Series
- Set start and end dates for the entire series
- Choose the interval type (day, week, month, year) and value
- Define data collection duration for each interval
- Creates a series of mosaics at regular intervals

#### 3. Output Settings

- **Output Directory**: Where to save processed mosaics
- **Stack Outputs**: Combine all time series mosaics into one file
- **Stack Format**: Choose between NetCDF or Multi-band GeoTIFF

#### 4. Advanced Settings

##### Mosaic Creation Options
- **Scene Sorting Method**: How to prioritize input scenes (valid_data, oldest, newest)
- **Mosaic Method**: How to merge overlapping areas (mean = average pixel values, first = use first valid pixel)
- **Required Bands**: Which spectral bands to include in the mosaic
- **No Data Threshold**: Threshold for handling no data values

##### Cloud Masking Options
- **OCM Batch Size**: Batch size for cloud mask inference
- **OCM Inference Data Type**: Data type for OCM inference
- **Cloud Cover Threshold**: Maximum cloud cover percentage for scene selection

#### 5. Processing Steps

The application performs the following steps for mosaic creation:

1. **Grid Processing**: For each selected grid tile:
   - Search and download available scenes for the specified time period
   - Apply cloud masking to each scene using OmniCloudMask
   - Create a mosaic by averaging overlapping pixels (when mosaic_method = "mean")
   - Apply geographic boundary masking if a vector file or bounding box was provided

2. **Time Series Creation** (if selected):
   - Generate mosaics at regular intervals throughout the specified time range
   - Each mosaic represents a composite of all available imagery for that interval

3. **Stacking** (if enabled):
   - Combine all time series mosaics into a single multi-dimensional file
   - NetCDF format preserves time and band dimensions for easier analysis
   - Multi-band GeoTIFF stacks all time steps as separate bands

#### 6. Output Files

The application generates the following output files:

- **Individual Mosaics**: GeoTIFF files for each time period with naming pattern:
  `{grid_id}_{date}.tif` or `{grid_id}_{date}_masked.tif` (if boundary masking was applied)
- **Stacked Outputs** (if enabled):
  - NetCDF: `{stack_filename}.nc` with dimensions for time, bands, y, and x
  - GeoTIFF: `{stack_filename}.tif` with each time step as a separate band

## Multicore Processing

The application supports multicore processing for batch operations, which can significantly speed up processing multiple files:

### How It Works

1. When you select **Folder Processing** mode in Cloud Masking, you'll see an option to **Use Multicore Processing**
2. If enabled, you can select the number of worker processes (parallel jobs) to use
3. The application will automatically distribute the workload across the specified number of CPU cores

### Performance Benefits

- **Linear Scaling**: Processing time typically decreases proportionally with more cores
- **Resource Utilization**: Makes efficient use of available CPU and memory resources
- **Progress Tracking**: The progress bar shows overall completion across all workers

### When to Use

- **Large Batch Jobs**: When processing many files or time periods
- **Multicore Systems**: Most beneficial on computers with 4+ CPU cores
- **Independent Files**: Each file/group is processed independently

## Application Structure

The application consists of multiple Python modules, each with a specific role:

- `⌂_Home.py`: Main application home page
- `1_☁️_Cloud_Masking.py`: Cloud masking interface
- `2_🧩_Mosaic_Builder.py`: Mosaic builder interface
- `file_utils.py`: File utility functions for finding and organizing raster files
- `processing.py`: Image processing functions for cloud masking
- `regularized_mosaic.py`: Functions for creating time series mosaics
- `visualization.py`: Visualization functions for displaying results

## Hardware Requirements

- **CPU**: Any modern CPU will work, but faster CPUs will process images more quickly
- **GPU**: CUDA-capable GPU recommended for faster processing
  - The app automatically detects and uses GPU if available
  - Falls back to CPU if no GPU is found
- **Memory**: At least 8GB RAM recommended, more for processing large images
- **Disk Space**: Sufficient space to store input and output imagery

## Troubleshooting

### Common Issues

- **File Format Errors**: Ensure input files are valid GeoTIFF files
- **Memory Errors**: When processing large files, try increasing the resampling factor
- **GPU Memory Issues**: Reduce batch size or patch size if experiencing CUDA out of memory errors
- **Missing Grid Tiles**: Make sure your vector file intersects with Sentinel-2 grid tiles
- **Empty Mosaics**: Try extending the time range or increasing the data collection duration

### Debug Tips

- Check the console output for error messages
- Inspect the intermediate visualization to verify cloud detection
- Try with a smaller test area if processing large regions

## Citation

If you use this application in your research, please cite both this application and the OmniCloudMask paper:

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

## License

This application is released under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- The OmniCloudMask team for their excellent cloud detection model
- The S2Mosaic developers for the Sentinel-2 mosaic creation tools
- The Streamlit team for their fantastic framework for building data applications
