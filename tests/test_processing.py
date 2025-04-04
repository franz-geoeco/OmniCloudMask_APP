# tests/test_processing.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from processing import process_multiband_file

@pytest.mark.parametrize("save_cloud_mask", [True, False])
def test_save_cloud_mask_option(save_cloud_mask):
    # Create mock objects
    mock_src = MagicMock()
    mock_src.read.return_value = np.zeros((10, 10))
    mock_src.meta = {'height': 10, 'width': 10, 'count': 3}
    mock_src.height = 10
    mock_src.width = 10
    mock_src.count = 3
    mock_src.nodata = None
    mock_src.crs = "EPSG:4326"
    mock_src.bounds.left = 0
    mock_src.bounds.bottom = 0
    mock_src.bounds.right = 1
    mock_src.bounds.top = 1
    mock_src.transform = [1, 0, 0, 0, 1, 0]
    
    # Mock predict_from_array
    mock_pred = np.zeros((1, 5, 5))
    
    # Mock file operations
    with patch('rasterio.open', return_value=mock_src), \
         patch('os.makedirs'), \
         patch('processing.predict_from_array', return_value=mock_pred), \
         patch('processing.plot_results'), \
         patch('streamlit.pyplot'), \
         patch('streamlit.success'), \
         patch('streamlit.write'), \
         patch('streamlit.info'):
        
        # Call function
        detection_options = {'save_cloud_mask': save_cloud_mask}
        result = process_multiband_file(
            'test.tif', 1, 2, 3, 'output_dir', 
            resampling_factor=2, device='cpu', 
            detection_options=detection_options
        )
        
        # Verify result
        assert result is True
        
        # Check if the file write operations were called correctly
        if save_cloud_mask:
            assert mock_src.meta.call_count == 2  # One for output, one for mask
        else:
            assert mock_src.meta.call_count == 1  # Only for output