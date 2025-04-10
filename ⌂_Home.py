import streamlit as st

st.set_page_config(
    page_title="Satellite Image Processing Suite",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Image Processing Suite")

st.markdown("""
This application provides tools for working with satellite imagery:

### Available Tools:

1. **☁️ Cloud Masking Tool**: Detect and mask clouds in satellite imagery using OmniCloudMask.
   - Works with Sentinel-2, Landsat, and other satellite imagery with RGB+NIR bands
   - Based on OmniCloudMask - a sensor-agnostic deep learning model

2. **🧩 Mosaic Builder**: Create and regularize mosaics from satellite imagery.
   - Build Sentinel-2 mosaics for specified grid areas and time ranges
   - Create time series of mosaics at regular intervals
   - Stack multiple mosaics for time series analysis

### How to Use:
- Select a tool from the sidebar on the left
- Follow the instructions on each page to process your data
- Results will be saved to your specified output locations

### About:
This application combines cloud masking and mosaic building capabilities to provide a comprehensive solution for satellite image processing. The cloud masking tool helps remove cloud interference from your imagery, while the mosaic builder allows you to create composite images over specific areas and time periods.

For more information on the underlying algorithms and methodologies, see the documentation pages for each tool.
""")

# About section
with st.expander("About the Developers", expanded=False):
    st.markdown("""
    #### Development Team
    
    This application was developed by a team of researchers and developers focused on improving satellite imagery processing workflows.
    
    #### References
    
    **OmniCloudMask**:
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

# Getting started section
with st.expander("Getting Started", expanded=True):
    st.markdown("""
    ### Quick Start Guide
    
    1. **Select a tool** from the sidebar on the left
    2. **Configure your inputs**:
       - For Cloud Masking: Select a file or folder with satellite imagery
       - For Mosaic Builder: Specify grid ID, time range, and other parameters
    3. **Adjust processing options** as needed
    4. **Run the processing** by clicking the start button
    5. **View results** and download the processed files
    
    ### Tips for Best Results
    
    - For large files, increase the resampling factor to speed up processing
    - GPU acceleration is automatically used when available
    - For batch processing, use the multicore processing option
    """)