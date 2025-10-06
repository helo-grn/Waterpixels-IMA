import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
from skimage import color, filters
import IPython
from skimage.transform import rescale

def viewimage(im, normalize=True,z=1,order=0,titre='',displayfilename=False):
    imin=im.copy().astype(np.float32)
    imin = rescale(imin, z, order=order)
    if normalize:
        imin-=imin.min()
        if imin.max()>0:
            imin/=imin.max()
    else:
        imin=imin.clip(0,255)/255 
    imin=(imin*255).astype(np.uint8)
    plt.imshow(imin)
    plt.axis('off')
    plt.title(titre)
    plt.show()
    
def derive(im, dir=1):
    s=im.shape
    im2= np.zeros(s)
    for i in range (s[0]):
        for j in range (s[1]):
            if dir==1:
                if j==0:
                    im2[i,j]=im[i,j+1]-im[i,j]
                elif j==s[1]-1:
                    im2[i,j]=im[i,j]-im[i,j-1]
                else:
                    im2[i,j]=(im[i,j+1]-im[i,j-1])/2
            else:
                if i==0:
                    im2[i,j]=im[i+1,j]-im[i,j]
                elif i==s[0]-1:
                    im2[i,j]=im[i,j]-im[i-1,j]
                else:
                    im2[i,j]=(im[i+1,j]-im[i-1,j])/2
    return im2

'''
def gradient(im):
    lab_img = color.rgb2lab(im)  
    viewimage(lab_img)
    gradient = filters.sobel(color.rgb2gray(lab_img))  
'''

def square_grid(im, step=0):
    w, h = im.shape
    print(h, w)
    if step == 0:
        step = min(h, w) // 10
    x = np.arange(0, h, step)
    y = np.arange(0, w, step)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    print(xv.shape, yv.shape)
    return xv, yv

def hexagonal_grid(im, step=0):
    w, h = im.shape
    print(h, w)
    if step == 0:
        step = min(h, w) // 5
    x1 = np.arange(0, h, step*np.sqrt(3))
    y1 = np.arange(0, w, step)
    xv1, yv1 = np.meshgrid(x1, y1, indexing='ij')
    print(xv1, yv1)
    x2 = np.arange(step*np.sqrt(3)//2, h, step*np.sqrt(3))
    y2 = np.arange(step//2, w, step)
    xv2, yv2 = np.meshgrid(x2, y2, indexing='ij')

    xv = np.concatenate([xv1.flatten(), xv2.flatten()])
    yv = np.concatenate([yv1.flatten(), yv2.flatten()])
    
    return xv, yv

def read_image(filename, grey):
    im = skio.imread(filename, as_gray=grey)
    return im

def test(n):
    if n == 0:
        # Test square grid
        im = read_image('Lake.png', True)
        xv, yv = square_grid(im)
        plt.plot(xv, yv, marker='o', color='k', linestyle='none')
        plt.axis('scaled')
        plt.show()
    if n == 2: 
        # Test hexagonal grid
        im = read_image('Lake.png', True)
        xv, yv = hexagonal_grid(im)
        plt.plot(xv, yv, marker='o', color='k', linestyle='none')
        plt.axis('scaled')
        plt.show()
    if n == 3:
        # Test gradient
        im = read_image('Lake.png', True)
        grad = derive(im)
        viewimage(grad)
    if n == 4:
        # Test view image
        im = read_image('Lake.png', False)
        viewimage(im)
    if n == 5:
        # Test gradient 2
        im = read_image('Lake.png', True)
        grad = gradient(im)
        viewimage(grad)
        
test(5)