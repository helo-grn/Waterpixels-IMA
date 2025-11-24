import numpy as np
from skimage import util
from queue import Queue
from skimage import exposure
from skimage.segmentation import mark_boundaries

def get_neighbors(p, shape):
    neighbors = []
    rows, cols = shape
    row, col = p    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:  #8-connectivity
                continue
            else:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    neighbors.append((r, c))
    return neighbors
    
def fast_watershed(im):
    """
    Compute watershed segmentation from scratch.
    """
    
    MASK, WSHED, INIT = -2, 0, -1
    current_label = 0
    queue = Queue()

    imout = np.full(im.shape, INIT, dtype=np.int32)
    imd = np.zeros(im.shape, dtype=np.int32)

    # Sort pixel coordinates by intensity
    idx = np.argsort(im.ravel())
    coords = np.column_stack(np.unravel_index(idx, im.shape))
    sorted_vals = im.ravel()[idx]
    unique_heights = np.unique(sorted_vals)

    for h in unique_heights:
        mask_pixels = []
        level_mask = np.argwhere(im == h)

        # Step 1: Mask pixels at level h
        for p in [tuple(pt) for pt in level_mask]:
            imout[p] = MASK
            for q in get_neighbors(p, im.shape):
                if imout[q] > 0 or imout[q] == WSHED:
                    imd[p] = 1
                    queue.put(p)
                    break
            mask_pixels.append(p)

        # Step 2: Flood propagation
        cur_dist = 1
        queue.put((-1, -1))
        while not queue.empty():
            p = queue.get()
            if p == (-1, -1):
                if queue.empty():
                    break
                queue.put((-1, -1))
                cur_dist += 1
                p = queue.get()
            for q in get_neighbors(p, im.shape):
                if imd[q] < cur_dist and (imout[q] > 0 or imout[q] == WSHED):
                    if imout[q] > 0:
                        if imout[p] == MASK or imout[p] == WSHED:
                            imout[p] = imout[q]
                        elif imout[p] != imout[q]:
                            imout[p] = WSHED
                    elif imout[p] == MASK:
                        imout[p] = WSHED
                elif imout[q] == MASK and imd[q] == 0:
                    imd[q] = cur_dist + 1
                    queue.put(q)

        # Step 3: Label new minima
        for p in mask_pixels:
            imd[p] = 0
            if imout[p] == MASK:
                current_label += 1
                queue.put(p)
                imout[p] = current_label
                while not queue.empty():
                    q = queue.get()
                    for r in get_neighbors(q, im.shape):
                        if imout[r] == MASK:
                            queue.put(r)
                            imout[r] = current_label
    
    return imout