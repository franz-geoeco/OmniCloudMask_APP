import numpy as np
import matplotlib.pyplot as plt

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