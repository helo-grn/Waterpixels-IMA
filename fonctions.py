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
from skimage.color import rgb2gray, rgb2lab
from skimage.morphology import area_opening, area_closing
from skimage.filters import sobel
from skimage.segmentation import watershed, mark_boundaries

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
    
    
    
def derive(im, dir=1):
    s=im.shape
    print(s)
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


def gradient(im):
    ''' 
    Compute the gradient magnitude of a gray image 
    '''
    gradient = filters.sobel(im)
    return gradient

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

# def lab_gradient(im):
#     lab = rgb2lab(im)
#     grad = np.zeros(im.shape[:2])
#     print(lab.shape)
#     # 3 channels
#     for c in range(3):
#         grad += sobel(lab[..., c]) ** 2
#     grad = np.sqrt(grad)
#     return grad

def lab_gradient(im):
    lab = rgb2lab(im)
    grad = sobel(lab[..., 0]) ** 2
    grad = np.sqrt(grad)
    return grad

def square_grid(im, sigma=0):
    w, h = im.shape[:2]
    grid_im = np.zeros((w, h))
    Q = []
    for x in range(sigma//2, h, sigma):
        for y in range(sigma//2, w, sigma):
            grid_im[y, x] = 1
            Q.append((y, x))
    return grid_im, Q

def hexagonal_grid(im, sigma):
    w, h = im.shape[:2] # (321, 481)
    grid_im = np.zeros((w, h))
    Q = []
    for x in range(sigma//2, h, int(np.sqrt(3)*sigma)):
        for y in range(sigma//2, w, sigma):
            grid_im[y, x] = 1
            Q.append((y, x))
    for x in range(sigma//2+int(np.sqrt(3)*sigma)//2, h, int(np.sqrt(3)*sigma)):
        for y in range(0, w, sigma):
            grid_im[y, x] = 1
            Q.append((y, x))
    return grid_im, Q

def d(p, q):
    return np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def distance(im, Q, sigma):
    w, h = im.shape[:2]
    dist_im = np.zeros((w, h))
    for row in range(0, h):
        for col in range(0, w):
            p = (col, row)
            min_dist = np.inf
            for q in Q:
                if d(p, q) - min_dist < 0:
                    min_dist = d(p, q)
            dist_im[col, row] = 2/sigma * min_dist
    return dist_im
            
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

def read_image(filename):
    im = skio.imread(filename)
    return im

def test(n):
    if n == 0:
        # Test square grid
        im = read_image('Lake.png')
        xv, yv = square_grid(im)
        plt.plot(xv, yv, marker='o', color='k', linestyle='none')
        plt.axis('scaled')
        plt.show()
    if n == 2: 
        # Test hexagonal grid
        im = read_image('Lake.png')
        xv, yv = hexagonal_grid(im, 40)
        plt.plot(xv, yv, marker='o', color='k', linestyle='none')
        plt.axis('scaled')
        plt.show()
    if n == 3:
        # Test view image
        im = read_image('Lake.png')
        viewimage(im)
    if n == 4: 
        # Test gray image read
        im = read_image('Lake.png')
        im_gray = rgb2gray(im)
        viewimage(im_gray, gray=True)
    if n == 5:
        # Test gradient
        im = read_image('Lake.png')
        grad = derive(im)
        viewimage(grad)
    if n == 6:
        # Test gradient 2
        im = read_image('Lake.png')
        #im_gray = rgb2gray(im)
        h, w, _ = im.shape
        sigma = min(h, w) // 10
        im_proc = preprocess_image(im, sigma)  
        print("a")
        grad = lab_gradient(im_proc)
        print("b")
        print(grad.shape)
        #grad2 = gradient(im_gray)
        viewimage(grad, gray=True)
        #viewimage(grad2, gray=True)
    if n == 7:
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
        dist_im = distance(im, Q, 40)
        viewimage(dist_im, gray=True)
        # Compute regularized gradient
        reg_im = gradient_regularization(dist_im, grad, 10)
        print(reg_im)
        viewimage(reg_im, gray=True)
        # Compute watershed transform
        markers = create_markers(im.shape, Q)
        labels = watershed(reg_im, markers=markers, watershed_line=True)
        viewimage(labels, gray=True)
        # Overlay boundaries on original image
        final = mark_boundaries(im, labels)
        viewimage(final)
    if n == 8:
        im = read_image('Lake.png')
        # Compute gradient
        sigma = 40
        im_proc = preprocess_image(im, sigma)  
        grad = lab_gradient(im_proc)
        print(grad)
        viewimage(grad, gray=True)
        # Compute hexagonal grid
        grid_im, Q = square_grid(im, 40)
        viewimage(grid_im, gray=True)
        # Compute distance to centers
        dist_im = distance(im, Q, 40)
        viewimage(dist_im, gray=True)
        # Compute regularized gradient
        reg_im = gradient_regularization(dist_im, grad, 10)
        print(reg_im)
        viewimage(reg_im, gray=True)
        # Compute watershed transform
        markers = create_markers(im.shape, Q)
        labels = watershed(reg_im, markers=markers, watershed_line=True)
        viewimage(labels, gray=True)
        # Overlay boundaries on original image
        final = mark_boundaries(im, labels)
        viewimage(final)
        
test(8)