import numpy as np

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

