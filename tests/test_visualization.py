import pytest
import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_results

def test_plot_results_binary_mask():
    """Test plot_results with a binary mask"""
    # Create test data
    rgb_stack = np.random.rand(3, 10, 10)  # 3-band RGB stack
    cloud_mask = np.zeros((10, 10))        # Binary mask
    cloud_mask[2:5, 2:5] = 1               # Add some "cloud" pixels
    
    # Call the function
    fig = plot_results(rgb_stack, cloud_mask)
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure)
    
    # Check that the figure has the right number of axes
    assert len(fig.axes) == 3
    
    # Clean up
    plt.close(fig)

def test_plot_results_with_3d_mask():
    """Test plot_results with a 3D mask that needs squeezing"""
    # Create test data
    rgb_stack = np.random.rand(3, 10, 10)       # 3-band RGB stack
    cloud_mask = np.zeros((1, 10, 10))          # 3D mask with single channel
    cloud_mask[0, 2:5, 2:5] = 1                 # Add some "cloud" pixels
    
    # Call the function
    fig = plot_results(rgb_stack, cloud_mask)
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure)
    
    # Check that the figure has the right number of axes
    assert len(fig.axes) == 3
    
    # Clean up
    plt.close(fig)

def test_plot_results_handles_normalization():
    """Test that plot_results handles image normalization correctly"""
    # Create test data with values outside 0-1 range
    rgb_stack = np.random.rand(3, 10, 10) * 5000  # Large values
    cloud_mask = np.zeros((10, 10))
    cloud_mask[2:5, 2:5] = 1
    
    # Call the function
    fig = plot_results(rgb_stack, cloud_mask)
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure)
    
    # Clean up
    plt.close(fig)

def test_plot_results_handles_nodata():
    """Test that plot_results handles nodata values correctly"""
    # Create test data with nodata values
    rgb_stack = np.random.rand(3, 10, 10)
    rgb_stack[:, 0:2, 0:2] = 0  # Add some nodata regions
    
    cloud_mask = np.zeros((10, 10))
    cloud_mask[2:5, 2:5] = 1
    
    # Call the function
    fig = plot_results(rgb_stack, cloud_mask)
    
    # Check that a figure was returned
    assert isinstance(fig, plt.Figure)
    
    # Clean up
    plt.close(fig)