import numpy as np
import platform
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
# necessite scikit-image 
from skimage import io as skio
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
    filename=tempfile.mktemp(titre+'.png')
    if displayfilename:
        print (filename)
    plt.imsave(filename, imin, cmap='gray')
    IPython.display.display(IPython.display.Image(filename))

def derive(im, dir=1):
    s=im.shape()
    im2= np.zeros((s,s))
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