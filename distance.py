import numpy as np
import grids as grd
import distance as dst
from skimage.color import rgb2gray

def d(p, q):
    """
    Compute the euclidian distance between two points.

    Parameters:
        p: First point
        q: Second point

    Returns:
        float: Distance
    """
    return np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def naive_distance(im, Q):
    w, h = im.shape[:2]
    dist_im = np.zeros((w, h))
    for row in range(0, h):
        for col in range(0, w):
            p = (col, row)
            min_dist = np.inf
            for q in Q:
                if d(p, q) - min_dist < 0:
                    min_dist = d(p, q)
            dist_im[col, row] = min_dist
    return dist_im


def chamfer_distance_5_7_11(binary_image):
    """
    Compute the chamfer distance transform using 5-7-11 mask.
    
    Parameters:
    binary_image : Binary image where markers are pixels equal to 1
    
    Returns:
    distance_map : Distance map where each pixel contains its distance to the nearest marker
    """
    # Initialize distance map with large values and zeroes for markers
    distance_map = np.where(binary_image == 1, 0, np.inf)
    
    height, width = binary_image.shape
    
    # Define the 5-7-11 chamfer mask
    # Format: (row_offset, col_offset, distance)
    # Forward pass mask (top-left to bottom-right)
    forward_mask = [
        (-2, -1, 11), (-2, 1, 11),
        (-1, -2, 11), (-1, -1, 7), (-1, 0, 5), (-1, 1, 7), (-1, 2, 11),
        (0, -1, 5)
    ]
    
    # Backward pass mask (bottom-right to top-left)
    backward_mask = [
        (0, 1, 5),
        (1, -2, 11), (1, -1, 7), (1, 0, 5), (1, 1, 7), (1, 2, 11),
        (2, -1, 11), (2, 1, 11)
    ]
    
    # Forward pass
    for i in range(height):
        for j in range(width):
            if distance_map[i, j] > 0:  # Skip markers
                min_dist = distance_map[i, j]
                
                for dy, dx, d in forward_mask:
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < height and 0 <= nj < width:
                        min_dist = min(min_dist, distance_map[ni, nj] + d)
                
                distance_map[i, j] = min_dist
    
    # Backward pass
    for i in range(height - 1, -1, -1):
        for j in range(width - 1, -1, -1):
            if distance_map[i, j] > 0:  # Skip markers
                min_dist = distance_map[i, j]
                
                for dy, dx, d in backward_mask:
                    ni, nj = i + dy, j + dx
                    if 0 <= ni < height and 0 <= nj < width:
                        min_dist = min(min_dist, distance_map[ni, nj] + d)
                
                distance_map[i, j] = min_dist
    
    return distance_map/5

def minima_gradient(im, grad, grid):
    """
    Computes the minima of the gradient of an image.
    First we compute the minima of the gradient g. Each minimum is a connected component.
    These minima are truncated along the grid, i.e. pixels which fall on the margins between cells is removed.
    Second, every cell of the grid serves to define a region of interest in the gradient image. 
    From this region we select a unique marker, chosen among the minima of g. If there are more than one, the one with the highest surface extinction value is used.

    Parameters:
        im: The input image (RGB).
        grad: The gradient of the input image.
        grid: The grid image.
    Returns:
        markers: The markers for the watershed transform.

    """
    im_shape = im.shape
    markers = np.zeros(im_shape[:2], dtype=np.int32)
    labeled_minima, num_minima = ndi.label(grad == ndi.minimum_filter(grad, size=3))
    
    for cell_label in np.unique(grid):
        if cell_label == 0:
            continue  # Skip background
        
        cell_mask = (grid == cell_label)
        cell_minima_labels = np.unique(labeled_minima[cell_mask & (labeled_minima > 0)])
        
        if len(cell_minima_labels) == 0:
            continue  # No minima in this cell
        
        max_intensity = -1
        selected_minimum = None
        
        for minimum_label in cell_minima_labels:
            minimum_mask = (labeled_minima == minimum_label)
            mean_intensity = np.mean(im[minimum_mask])
            
            if mean_intensity > max_intensity:
                max_intensity = mean_intensity
                selected_minimum = minimum_label
        
        if selected_minimum is not None:
            markers[labeled_minima == selected_minimum] = cell_label
    
    return markers

