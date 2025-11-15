import numpy as np
from skimage.segmentation import mark_boundaries
from scipy.ndimage import gaussian_filter

from gradient import *
from grids import *
from distance import *
from watershed import *
from fonctions import *

def test(n):
    if n == 0:
        im = read_image('Lake.png')
        # im = rgb2gray(im)
        # im = np.round(im, 3)
        reg_im = np.load('reg_im.npy')
        reg_im = gaussian_filter(reg_im, sigma=3)
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