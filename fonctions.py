import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.morphology import area_opening, area_closing
from skimage.filters import sobel
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

def viewimage(im, gray=False, normalize=True,z=1,order=0,titre='',displayfilename=False):
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

def waterpixels(im, gradient_method='naive', grid='hexagonal'):
    im = read_image('Lake.png')
    # Compute gradient
    sigma = 40
    im_proc = preprocess_image(im, sigma)  
    if gradient_method == 'lab':
        grad = lab_gradient(im_proc)
    else:
        grad = gradient(im_proc)
        
    viewimage(grad, gray=True)
    
    # Compute grid
    if grid == 'square':
        grid_im, Q = square_grid(im, 40)
    else:
        grid_im, Q = hexagonal_grid(im, 40)
        
    viewimage(grid_im, gray=True)
    
    # Compute distance to centers
    dist_im = naive_distance(im, Q, 40)
    viewimage(dist_im, gray=True)
    
    # Compute regularized gradient
    reg_im = gradient_regularization(dist_im, grad, 10)
    viewimage(reg_im, gray=True)
    
    # Compute watershed transform
    markers = create_markers(im.shape, Q)
    labels = watershed(reg_im, markers=markers, watershed_line=True)
    viewimage(labels, gray=True)
    
    # Overlay boundaries on original image
    final = mark_boundaries(im, labels)
    viewimage(final)

def test(n):
    if n == 0:
        im = read_image('Lake.png')
        # im = rgb2gray(im)
        # im = np.round(im, 3)
        reg_im = np.load('reg_im.npy')
        reg_im = gaussian_filter(reg_im, sigma=2)
        reg_im = reg_im * 10
        reg_im = np.round(reg_im)
        reg_im = np.round(reg_im / 40) * 40
        labels = fast_watershed2(reg_im)
        print(labels.shape)
        final = mark_boundaries(im, labels)
        viewimage(final)
    if n == 1:
        # Trying with our own distance transform
        im = read_image('Lake.png')
        # Compute gradient
        sigma = 40
        im_proc = preprocess_image(im, sigma)  
        grad = lab_gradient(im_proc)
        print(grad)
        viewimage(grad, gray=True)
        # Compute hexagonal grid
        grid_im, Q = hexagonal_grid(im, 40)
        viewimage(grid_im, gray=True)
        # Compute distance to centers
        dist_im = chamfer_distance_transform(grid_im)
        viewimage(dist_im, gray=True)

#faire un truc d'histogramme pour découper en 10 niveaux de gris