from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
import numpy as np

def derive(im, dir=0):
    """
    Computes the derivative of an image using the finite difference approximation.

    Parameters:
        im: The input image (RGB).
        dir: Axis alongside which the derivative is computed (0 = vertical, 1 = horizontal).
        
    Returns:
        im2: The derivative alongside the chosen axis.
    """
    im = rgb2gray(im)
    h, w = im.shape
    im2= np.zeros_like(im)
    if dir == 0:
        for i in range(h):
            im2[i,0] = im[i,1] - im[i,0]
            im2[i,w-1] = im[i,w-1] - im[i,w-2]
            for j in range(1, w-1):
                im2[i,j] = (im[i,j+1] - im[i,j-1])/2
    else:
        for j in range(w):
            im2[0,j] = im[1,j] - im[0,j]
            im2[h-1,j] = im[h-1,j] - im[h-2,j]
            for i in range(1, h-1):
                im2[i,j] = (im[i+1,j] - im[i-1,j])/2
    return im2

def gradient(im):
    """
    Computes the gradient of an image using the finite difference approximation.

    Parameters:
        im: The input image (RGB).
        
    Returns:
        grad: The gradient of the image
    """
    dx = derive(im, dir=0)
    dy = derive(im, dir=1)
    grad = np.sqrt(dx**2 + dy**2)
    return grad

def lab_gradient(im):
    """
    Computes the gradient of an image.

    Parameters:
        im: The input image (RGB).

    Returns:
        grad: The gradient on the perceptual lightness.
    """
    lab_im = rgb2lab(im)
    grad = sobel(lab_im[..., 0]) ** 2
    grad = np.sqrt(grad)
    return grad