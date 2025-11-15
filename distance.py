import numpy as np

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
    -----------
    binary_image : numpy.ndarray
        Binary image where markers are pixels equal to 1
    
    Returns:
    --------
    distance_map : numpy.ndarray
        Distance map where each pixel contains its distance to nearest marker
    """
    # Initialize distance map with large values
    # Set markers (pixels == 1) to distance 0
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
