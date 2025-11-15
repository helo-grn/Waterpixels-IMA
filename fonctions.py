import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.transform import rescale
from skimage.morphology import area_opening, area_closing
from skimage.segmentation import watershed, mark_boundaries
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt

from gradient import *
from grids import *
from distance import *
from watershed import *

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
    print(imin.shape)
    print(imin.dtype)
    print(imin[0, 0])
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
            
def gradient_regularization(dist_im, grad_im, k):
    reg_im = grad_im + k * dist_im
    return reg_im

def create_markers(im_shape, Q):
    markers = np.zeros(im_shape[:2], dtype=np.int32)
    for idx, (y, x) in enumerate(Q):
        # Ensure coordinates are within image bounds
        if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
            markers[y, x] = idx + 1  # labels start from 1
    return markers

def waterpixels(image_name, nb_pixels, gradient_method='naive', grid='hexagonal', distance_alg='Chamfer', watershed_alg='skimage'):
    """
    Assembles all the functions to compute the waterpixels of a given image and then display the boundaries on top of the original image.

    Parameters:
        image_name: The filename of the input image (RGB).
        nb_pixels: The number of waterpixels to compute alongside the smallest edge of the image.
        gradient_method: Can be _naive_ (discrete approximation of the derivative) or _lab_ (Sobel filter on the lightness component in the CIELAB color space).
        grid: The approximate shape of the resulting waterpixels. Can be _hexagonal_ or _square_.
        distance_alg: The algorithm used to compute the distance map. Can be _naive_ or _Chamfer_.
        watershed_alg: The algorithm used to compute the watershed transform. Can be _skimage_ or _fast_.
    
    Returns:
        seg: The segmentation of the given image using waterpixels.
    """
    im = read_image(image_name)
    l = min(im.shape[0], im.shape[1])
    # Compute gradient
    sigma = np.round(l // nb_pixels)
    im_proc = preprocess_image(im, sigma)  
    if gradient_method == 'lab':
        grad = lab_gradient(im_proc)
    else:
        grad = gradient(im_proc)
        
    viewimage(grad, gray=True)
    
    # Compute grid
    if grid == 'square':
        grid_im, Q = square_grid(im, sigma)
    else:
        grid_im, Q = hexagonal_grid(im, sigma)
        
    viewimage(grid_im, gray=True)
    
    # Compute distance to centers
    if distance_alg == 'naive':
        dist_im = naive_distance(im, Q)
    else:
        dist_im = chamfer_distance_5_7_11(grid_im)
    viewimage(dist_im, gray=True)
    
    # Compute regularized gradient
    reg_im = gradient_regularization(dist_im, grad, 0.7)
    viewimage(reg_im, gray=True)
    
    # Compute watershed transform
    markers = create_markers(im.shape, Q)
    
    if watershed_alg == 'fast':
        reg_im = gaussian_filter(reg_im, sigma=3)
        reg_im = reg_im * 10
        reg_im = np.round(reg_im)
        reg_im = np.round(reg_im / 40) * 40
        labels = fast_watershed2(reg_im)
    else:
        labels = watershed(reg_im, markers=markers, watershed_line=True)
    
    viewimage(labels, gray=True)
    
    # Overlay boundaries on original image
    seg = mark_boundaries(im, labels)

    return seg

#faire un truc d'histogramme pour découper en 10 niveaux de gris

seg = waterpixels('Lake.png', 8, gradient_method='lab', grid='hexagonal', distance_alg='Chamfer', watershed_alg='skimage')
viewimage(seg)