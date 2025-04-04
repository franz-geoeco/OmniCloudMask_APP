import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

@patch('cloud_detection.predict_from_array')
def test_binary_mask_enforced(mock_predict):
    """
    Test that export_confidence=False and softmax_output=False are enforced
    to ensure we always get binary mask outputs
    """
    # Setup the mock
    mock_predict.return_value = np.zeros((1, 5, 5))
    
    # Import locally in the test to avoid import errors
    from processing import process_multiband_file
    
    # Create a mock for the rest of the function to avoid execution
    with patch('os.makedirs'), \
         patch('rasterio.open', side_effect=Exception("Test shortcut")), \
         patch('streamlit.error'):
        
        try:
            # Call with export_confidence=True in options (should be overridden)
            process_multiband_file(
                'dummy.tif', 1, 2, 3, 'output', 
                detection_options={
                    'export_confidence': True,  # This should be overridden
                    'softmax_output': True,     # This should be overridden
                    'batch_size': 2
                }
            )
        except Exception:
            # Expected to raise exception due to our mock
            pass
        
    # Check if predict_from_array was called and with what parameters
    # It should include export_confidence=False and softmax_output=False
    # regardless of what was passed in detection_options
    mock_predict.assert_called()
    
    # Get the keyword arguments passed to predict_from_array
    call_kwargs = mock_predict.call_args[1]
    
    # Verify the options were properly enforced
    assert call_kwargs.get('export_confidence') is False
    assert call_kwargs.get('softmax_output') is False
    assert call_kwargs.get('batch_size') == 2  # Other options should be preserved

# We need to mock at the module level
@patch('cloud_detection.predict_from_array')
def test_save_cloud_mask_option(mock_predict):
    """Test that the save_cloud_mask option is extracted and not passed to predict_from_array"""
    # Setup the mock
    mock_predict.return_value = np.zeros((1, 5, 5))
    
    # Import locally in the test
    from processing import process_multiband_file
    
    # Create a mock for rasterio open that returns the same mock object for all calls
    mock_src = MagicMock()
    mock_src.read.return_value = np.zeros((10, 10))
    mock_src.meta = {'height': 10, 'width': 10, 'count': 3}
    mock_src.height = 10
    mock_src.width = 10
    mock_src.count = 3
    mock_src.nodata = None
    
    # Mock bounds and transform
    mock_bounds = MagicMock()
    mock_bounds.left = 0
    mock_bounds.bottom = 0
    mock_bounds.right = 1
    mock_bounds.top = 1
    mock_src.bounds = mock_bounds
    
    # Set CRS
    mock_src.crs = "EPSG:4326"
    
    # Create mock transform
    from rasterio.transform import Affine
    mock_src.transform = Affine(1, 0, 0, 0, 1, 0)
    
    # Create a dummy array for np.where to use
    dummy_arrays = [np.array([1, 2]), np.array([1, 2])]
    
    # Use a more controlled approach to patching
    with patch('os.makedirs'), \
         patch('rasterio.open', return_value=mock_src), \
         patch('processing.plot_results'), \
         patch('numpy.where', return_value=dummy_arrays), \
         patch('numpy.any', return_value=np.ones((10, 10), dtype=bool)), \
         patch('numpy.zeros'), \
         patch('numpy.full'), \
         patch('numpy.stack', return_value=np.zeros((3, 5, 5))), \
         patch('rasterio.warp.reproject'), \
         patch('streamlit.pyplot'), \
         patch('streamlit.success'), \
         patch('streamlit.write'), \
         patch('streamlit.info'), \
         patch('streamlit.error'):
        
        try:
            # Call with save_cloud_mask in options
            process_multiband_file(
                'dummy.tif', 1, 2, 3, 'output', 
                detection_options={
                    'save_cloud_mask': True,
                    'batch_size': 2
                }
            )
        except Exception as e:
            print(f"Exception during test: {str(e)}")
            pass
    
    # Verify predict_from_array was called and save_cloud_mask was NOT included
    mock_predict.assert_called()
    
    # Check the kwargs to make sure save_cloud_mask wasn't passed
    call_kwargs = mock_predict.call_args[1]
    assert 'save_cloud_mask' not in call_kwargs
    assert 'batch_size' in call_kwargs

@patch('cloud_detection.predict_from_array')
def test_torch_dtype_conversion(mock_predict):
    """Test that string dtype values are converted to torch dtypes"""
    import torch
    
    # Setup the mock
    mock_predict.return_value = np.zeros((1, 5, 5))
    
    # Import the function to test
    from processing import process_multiband_file
    
    # Mock just enough to get to the predict_from_array call
    with patch('rasterio.open', side_effect=Exception("Test shortcut")), \
         patch('streamlit.error'):
        
        try:
            # Test with string dtype
            process_multiband_file(
                'dummy.tif', 1, 2, 3, 'output', 
                detection_options={
                    'inference_dtype': 'float32'
                }
            )
        except Exception:
            pass
    
    # Verify predict_from_array was called
    mock_predict.assert_called()
    
    # Check that dtype was converted to torch.float32
    call_kwargs = mock_predict.call_args[1]
    assert call_kwargs.get('inference_dtype') is torch.float32
