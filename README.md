# OMNI Cloud Masking Application

A Streamlit application for detecting and masking clouds in satellite imagery using OmniCloudMask.

## About OmniCloudMask

OmniCloudMask (OCM) is a sensor-agnostic deep learning model that segments clouds and cloud shadows. It demonstrates robust state-of-the-art performance across various satellite platforms when classifying clear, cloud, and shadow classes, with balanced overall accuracy values across:
- **Landsat**: 91.5% clear, 91.5% cloud, and 75.2% shadow
- **Sentinel-2**: 92.2% clear, 91.2% cloud, and 80.5% shadow
- **PlanetScope**: 96.9% clear, 98.8% cloud, and 97.4% shadow

**GitHub**: [https://github.com/DPIRD-DMA/OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)

**Paper**: [Training sensor-agnostic deep learning models for remote sensing: Achieving state-of-the-art cloud and cloud shadow identification with OmniCloudMask](https://www.sciencedirect.com/science/article/pii/S0034425725000987)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cloud-masking-app.git
cd cloud-masking-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install OmniCloudMask:
```bash
pip install omnicloudmask
```

## Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser, typically at http://localhost:8501.

## Features

- Process single multiband GeoTIFF files or folders of single-band files
- Automatic detection of file groups by date/tile
- Customizable resampling for faster processing
- GPU detection and utilization when available
- Advanced cloud detection parameters
- Visualization of results
- Batch processing

## Application Structure

- `app.py`: Main Streamlit application
- `file_utils.py`: File utility functions
- `processing.py`: Image processing functions
- `cloud_detection.py`: Cloud detection module
- `visualization.py`: Visualization functions

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