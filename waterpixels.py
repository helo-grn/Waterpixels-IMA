import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.transform import rescale
from skimage.morphology import area_opening, area_closing
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter

from gradient import *
from grids import *
from distance import *
from watershed import *
from evaluation import *

def read_image(filename):
    im = skio.imread(filename)
    return im

def viewimage(im, gray=False, normalize=True,z=1,order=0,titre=''):
    imin=im.copy().astype(np.float32)
    imin = rescale(imin, z, order=order)
    if normalize:
        imin-=imin.min()
        if imin.max()>0:
            imin/=imin.max()
    else:
        imin=imin.clip(0,255)/255 
    imin=(imin*255).astype(np.uint8)
    if gray == True:
        plt.imshow(imin, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(imin)
    plt.axis('off')
    plt.title(titre)
    plt.show()

def preprocess_image(im, sigma):
    area = int((sigma ** 2) / 16)
    # If image is color, apply per channel
    if im.ndim == 3:
        im_proc = np.zeros_like(im)
        for c in range(im.shape[2]):
            im_proc[..., c] = area_opening(im[..., c], area)
            im_proc[..., c] = area_closing(im_proc[..., c], area)
    else:
        im_proc = area_opening(im, area)
        im_proc = area_closing(im_proc, area)
    return im_proc

def create_markers(im_shape, Q):
    markers = np.zeros(im_shape[:2], dtype=np.int32)
    for idx, (y, x) in enumerate(Q):
        # Ensure coordinates are within image bounds
        if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
            markers[y, x] = idx + 1  # labels start from 1
    return markers

def waterpixels(im, nb_pixels, k=0.7, gradient_method='naive', grid='hexagonal', markers='centers', distance_alg='Chamfer', watershed_alg='skimage'):
    """
    Assembles all the functions to compute the waterpixels of a given image and then display the boundaries on top of the original image.

    Parameters:
        im: The input image (RGB).
        nb_pixels: The number of waterpixels to compute alongside the smallest edge of the image.
        k: The regularization parameter for the gradient regularization step.
        gradient_method: Can be _naive_ (discrete approximation of the derivative) or _lab_ (Sobel filter on the lightness component in the CIELAB color space).
        grid: The approximate shape of the resulting waterpixels. Can be _hexagonal_ or _square_.
        markers: The choice of markers for the watershed transform. Can be _centers_ (centers of the grid cells) or _minima_ (minima of the gradient within each cell).
        distance_alg: The algorithm used to compute the distance map. Can be _naive_ or _chanfrein_.
        watershed_alg: The algorithm used to compute the watershed transform. Can be _skimage_ or _fast_.
    
    Returns:
        seg: The segmentation of the given image using waterpixels.
    """
    
    # Make sure that the parameters are valid
    assert gradient_method in ['naive', 'lab'], "Invalid gradient method. Choose 'naive' or 'lab'."
    assert grid in ['square', 'hexagonal'], "Invalid grid. Choose 'square' or 'hexagonal'."
    assert markers in ['centers', 'minima'], "Invalid markers. Choose 'centers' or 'minima'."
    assert distance_alg in ['naive', 'chanfrein'], "Invalid distance algorithm. Choose 'naive' or 'chanfrein'."
    assert watershed_alg in ['fast', 'skimage'], "Invalid watershed algorithm. Choose 'fast' or 'skimage'."

    
    # ¨Preprocess image
    l = min(im.shape[0], im.shape[1])
    sigma = np.round(l // nb_pixels)
    im_proc = preprocess_image(im, sigma)  
    viewimage(im_proc, gray=False, titre='Preprocessed image')
    
    # Compute gradient
    if gradient_method == 'lab':
        grad = lab_gradient(im_proc)
    elif gradient_method == 'naive':
        grad = gradient(im_proc)
    viewimage(grad, gray=True, titre='Gradient image')
    
    # Compute grid
    if grid == 'square':
        grid_im, Q = square_grid(im, sigma)
    elif grid == 'hexagonal':
        grid_im, Q = hexagonal_grid(im, sigma)
    viewimage(grid_im, gray=True, titre='Grid image')
    
    # Compute distance to centers
    if distance_alg == 'naive':
        dist_im = naive_distance(im, Q)
    elif distance_alg == 'chanfrein':
        dist_im = chamfer_distance_5_7_11(grid_im)
    viewimage(dist_im, gray=True, titre='Distance map')
    
    # Display distance map with grid centers
    if markers == 'centers':
        color_dist = np.zeros(im.shape)
        color_dist[:, :, 0] = dist_im
        color_dist[:, :, 1] = dist_im
        color_dist[:, :, 2] = dist_im
        color_dist[np.where(grid_im>0)] = [0, np.max(dist_im), 0]
        viewimage(color_dist, gray=False, titre='Distance map with grid centers')
        
    # Compute markers
    if markers == 'minima':
        a = fast_watershed(dist_im)
        b = segmentation_borders(a)
        a[np.where(b==1)] = 0
        minima = minima_gradient(im, grad, a, 3)
        minima_with_markers = np.zeros(im.shape)
        minima_with_markers[np.where(minima>0)] = [0, 255, 0]
        minima_with_markers[np.where(b==1)] = [255, 255, 255]
        viewimage(minima_with_markers, gray=False, titre='Gradient minima in each grid cell')
        
        minima[minima>0] = 1
        dist_im = chamfer_distance_5_7_11(minima)
        dist_with_markers = np.zeros(im.shape)
        dist_with_markers[:, :, 0] = dist_im
        dist_with_markers[:, :, 1] = dist_im
        dist_with_markers[:, :, 2] = dist_im
        dist_with_markers[np.where(minima>0)] = [0, np.max(dist_im), 0]
        viewimage(dist_with_markers, gray=False, titre='Distance map with minima markers')
    elif markers == 'centers':
        markers = create_markers(im.shape, Q)
    
    # Compute regularized gradient
    reg_im = gradient_regularization(dist_im, grad, k)
    viewimage(reg_im, gray=True, titre='Regularized gradient image')
    
    # Compute watershed transform
    if watershed_alg == 'fast':
        reg_im = gaussian_filter(reg_im, sigma=3)
        reg_im = reg_im * 10
        reg_im = np.round(reg_im)
        reg_im = np.round(reg_im / sigma) * sigma
        labels = fast_watershed(reg_im)
    elif watershed_alg == 'skimage':
        labels = watershed(reg_im, markers=markers, watershed_line=True)
    
    return labels