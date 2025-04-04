import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

def test_binary_mask_enforced():
    """
    Test that export_confidence=False and softmax_output=False are enforced
    to ensure we always get binary mask outputs
    """
    # Create a mock predict_from_array function to capture the arguments
    with patch('processing.predict_from_array') as mock_predict:
        # Setup the mock
        mock_predict.return_value = np.zeros((1, 5, 5))
        
        # Import locally in the test to avoid import errors
        from processing import process_multiband_file
        
        # Create a mock for the rest of the function to avoid execution
        with patch('os.makedirs'), \
             patch('rasterio.open', side_effect=Exception("Test shortcut")):
            
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

def test_save_cloud_mask_option():
    """Test that the save_cloud_mask option is extracted and not passed to predict_from_array"""
    # Create a mock predict_from_array function
    with patch('processing.predict_from_array') as mock_predict:
        # Setup the mock
        mock_predict.return_value = np.zeros((1, 5, 5))
        
        # Import locally in the test
        from processing import process_multiband_file
        
        # Create a mock for rasterio open that allows us to get further in the function
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
        mock_transform = MagicMock()
        mock_src.transform = mock_transform
        
        # Mock np.where to avoid actual mask computation
        with patch('os.makedirs'), \
             patch('rasterio.open', return_value=mock_src), \
             patch('numpy.any', return_value=np.zeros((10, 10), dtype=bool)), \
             patch('numpy.where', return_value=(np.array([1, 2]), np.array([1, 2]))), \
             patch('rasterio.transform.from_bounds'), \
             patch('rasterio.transform.from_origin'), \
             patch('processing.plot_results'), \
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
                # The function will likely raise an exception due to incomplete mocking
                pass
            
            # Verify predict_from_array was called and save_cloud_mask was NOT included
            mock_predict.assert_called()
            
            call_kwargs = mock_predict.call_args[1]
            assert 'save_cloud_mask' not in call_kwargs
            assert call_kwargs.get('batch_size') == 2

def test_torch_dtype_conversion():
    """Test that string dtype values are converted to torch dtypes"""
    import torch
    
    with patch('processing.predict_from_array') as mock_predict:
        mock_predict.return_value = np.zeros((1, 5, 5))
        
        from processing import process_multiband_file
        
        with patch('rasterio.open', side_effect=Exception("Test shortcut")):
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
            
            # Check that dtype was converted to torch.float32
            call_kwargs = mock_predict.call_args[1]
            assert call_kwargs.get('inference_dtype') is torch.float32
            
            # Reset mock for next test
            mock_predict.reset_mock()
            
            try:
                # Test with string dtype (float16)
                process_multiband_file(
                    'dummy.tif', 1, 2, 3, 'output', 
                    detection_options={
                        'inference_dtype': 'float16'
                    }
                )
            except Exception:
                pass
            
            # Check that dtype was converted to torch.float16
            call_kwargs = mock_predict.call_args[1]
            assert call_kwargs.get('inference_dtype') is torch.float16