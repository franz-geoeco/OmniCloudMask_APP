import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st

def plot_results(rgb_stack, cloud_mask):
    """
    Visualizes the RGB image, cloud mask, and overlay
    
    Parameters:
    rgb_stack (np.array): Stack of RGB bands in shape (3, height, width)
    cloud_mask (np.array): Cloud mask array in shape (height, width)
    
    Returns:
    matplotlib.figure.Figure: Figure object containing the visualization
    """
    # First, reshape the arrays properly for plotting
    # For RGB, transpose from (3, height, width) to (height, width, 3)
    rgb_image = np.transpose(rgb_stack, (1, 2, 0))
    
    # Normalize RGB image for display (false color)
    # Use percentile-based scaling to improve visualization
    p_low, p_high = np.percentile(rgb_image[rgb_image > 0], (2, 98))
    rgb_norm = np.clip((rgb_image - p_low) / (p_high - p_low), 0, 1)
    
    # For the mask, ensure it's 2D
    if len(cloud_mask.shape) > 2:
        mask = np.squeeze(cloud_mask)  # Remove single dimensions
    else:
        mask = cloud_mask
    
    # Standard binary mask visualization (always use this)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot RGB image
    axes[0].imshow(rgb_norm)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Plot mask
    axes[1].imshow(mask, cmap='viridis')
    axes[1].set_title('Cloud Mask')
    axes[1].axis('off')
    
    # Create a custom colormap for overlay with transparency
    cmap = plt.cm.Reds
    cmap.set_bad(alpha=0)  # Set transparency for masked values
    
    # Plot overlay: RGB image with mask overlay
    axes[2].imshow(rgb_norm)
    # Make a masked version where only values > 0 are shown
    masked_data = np.ma.masked_where(mask <= 0, mask)
    axes[2].imshow(masked_data, cmap=cmap, alpha=0.7)
    axes[2].set_title('RGB with Cloud Mask Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig

def calculate_cloud_statistics(cloud_mask):
    """
    Calculate statistics about cloud coverage
    
    Parameters:
    cloud_mask (np.array): Cloud mask array (height, width)
    
    Returns:
    dict: Dictionary with cloud statistics
    """
    # Ensure mask is properly formed
    if cloud_mask is None or cloud_mask.size == 0:
        return {"error": "Invalid cloud mask"}
    
    # Get counts of each class (0=clear, 1=thick cloud, 2=thin cloud, 3=shadow)
    total_pixels = cloud_mask.size
    valid_pixels = np.sum(cloud_mask != 255)  # Exclude no-data pixels
    
    # Calculate percentages for each class
    if valid_pixels > 0:
        clear_pixels = np.sum(cloud_mask == 0)
        thick_cloud_pixels = np.sum(cloud_mask == 1)
        thin_cloud_pixels = np.sum(cloud_mask == 2)
        shadow_pixels = np.sum(cloud_mask == 3)
        
        stats = {
            "clear_percent": (clear_pixels / valid_pixels) * 100,
            "thick_cloud_percent": (thick_cloud_pixels / valid_pixels) * 100,
            "thin_cloud_percent": (thin_cloud_pixels / valid_pixels) * 100,
            "shadow_percent": (shadow_pixels / valid_pixels) * 100,
            "total_cloud_percent": ((thick_cloud_pixels + thin_cloud_pixels + shadow_pixels) / valid_pixels) * 100,
            "valid_coverage_percent": (valid_pixels / total_pixels) * 100
        }
    else:
        stats = {
            "clear_percent": 0,
            "thick_cloud_percent": 0,
            "thin_cloud_percent": 0, 
            "shadow_percent": 0,
            "total_cloud_percent": 0,
            "valid_coverage_percent": 0
        }
    
    return stats

def display_cloud_statistics(cloud_mask):
    """
    Display cloud statistics in the Streamlit interface
    
    Parameters:
    cloud_mask (np.array): Cloud mask array
    """
    stats = calculate_cloud_statistics(cloud_mask)
    
    if "error" in stats:
        st.error(stats["error"])
        return
    
    st.subheader("Cloud Coverage Statistics")
    
    # Create columns for statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Clear Sky", f"{stats['clear_percent']:.2f}%")
        st.metric("Thick Clouds", f"{stats['thick_cloud_percent']:.2f}%")
    
    with col2:
        st.metric("Thin Clouds", f"{stats['thin_cloud_percent']:.2f}%")
        st.metric("Cloud Shadows", f"{stats['shadow_percent']:.2f}%")
    
    # Display total cloud coverage with a progress bar
    st.subheader("Total Cloud Coverage")
    st.progress(min(stats['total_cloud_percent'] / 100, 1.0))
    st.caption(f"{stats['total_cloud_percent']:.2f}% of image covered by clouds/shadows")
