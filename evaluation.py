import numpy as np
from scipy.ndimage import distance_transform_cdt, center_of_mass
import skimage.morphology as morpho


def morphological_gradient(partition):
    """
    Computes the morphological gradient of an image partition using a 4-neighborhood (cross-shaped structuring element).
    """
    
    structure_4n = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=bool)
    
    dilated = morpho.dilation(partition, footprint=structure_4n)
    eroded = morpho.erosion(partition, footprint=structure_4n)
    
    gradient = dilated - eroded
    
    return gradient

def add_borders(partition):
    """
    Adds a one-pixel-wide border around the image partition.
    """
    bordered = partition.copy()
    bordered[0, :] = 1
    bordered[-1, :] = 1
    bordered[:, 0] = 1
    bordered[:, -1] = 1
    return bordered

# -----------------------------------------------------------
# 1. Boundary Recall (BR)
# -----------------------------------------------------------
def boundary_recall(Sc, Sb, gt):
    """
    BR = percentage of GT contour pixels falling within <3 px L1 of C
    C = Sc ∪ Sb
    """
    C = (Sc | Sb).astype(np.uint8)

    # compute L1 (city-block) distance to C
    # distance_transform_cdt gives city-block distances
    dist = distance_transform_cdt(1 - C, metric='taxicab')

    close_to_contour = (gt == 1) & (dist < 3)
    return close_to_contour.sum() / gt.sum()


# -----------------------------------------------------------
# 2. Contour Density (CD)
# -----------------------------------------------------------
def contour_density(Sc, Sb, image_shape):
    """
    CD = (|Sc|/2 + |Sb|) / |D|
    """
    h, w = image_shape
    Sc_count = Sc.sum()
    Sb_count = Sb.sum()
    return (Sc_count / 2 + Sb_count) / (h * w)


# -----------------------------------------------------------
# 3. Average Mismatch Factor (MF)
# -----------------------------------------------------------
def extract_superpixels(partition):
    labels = np.unique(partition)
    return [ (l, np.argwhere(partition==l)) for l in labels ]

def center_superpixel(coords):
    """
    coords: N x 2 array of (row, col)
    return coords translated so barycenter is at (0,0)
    """
    cy, cx = coords.mean(axis=0)
    return coords - np.array([cy, cx])

def rasterize(coords):
    """
    Convert centered coords (float) to integer pixel set.
    """
    coords_rounded = np.round(coords).astype(int)
    return set(map(tuple, coords_rounded))

def compute_average_shape(superpixels):
    """
    Implements Appendix of the paper exactly:
    - Center all superpixels
    - Build S(x) = number of SP covering each x
    - Find highest threshold t0 such that |St| >= mean area
    """
    # 1. Extract centered shapes as sets
    centered_sets = []
    areas = []

    all_points = []

    for label, coords in superpixels:
        c = center_superpixel(coords)
        s = rasterize(c)
        centered_sets.append(s)
        areas.append(len(s))
        all_points.extend(list(s))

    areas = np.array(areas)
    mu_A = areas.mean()

    # Build S(x): count how many superpixels include point x
    # Use dict because coordinates can be large
    S = {}
    for s in centered_sets:
        for p in s:
            S[p] = S.get(p, 0) + 1

    # Determine thresholds
    values = sorted(set(S.values()), reverse=True)

    # Find highest t0 such that |St| >= mu_A
    for t in values:
        St = {p for p, v in S.items() if v >= t}
        if len(St) >= mu_A:
            return St  # this is the average centered shape

    # fallback
    return set()


def mismatch_factor(A, B):
    """
    m_f(A, B) = 1 - |A ∩ B| / |A ∪ B|
    """
    inter = len(A & B)
    union = len(A | B)
    if union == 0:
        return 0
    return 1 - inter / union


def average_mismatch_factor(partition):
    """
    MF = (1/N) sum_i m_f(s*_i, ŝ*)
    """
    superpixels = extract_superpixels(partition)
    centered_sets = []

    for _, coords in superpixels:
        c = center_superpixel(coords)
        s = rasterize(c)
        centered_sets.append(s)

    avg_shape = compute_average_shape(superpixels)

    mfs = [ mismatch_factor(s, avg_shape) for s in centered_sets ]
    return np.mean(mfs)


# -----------------------------------------------------------
# Complete evaluation wrapper
# -----------------------------------------------------------
def evaluate_waterpixels_measures(partition, gt):
    """
    partition: 2D array of superpixel labels
    gt: ground-truth contour mask (0/1)
    """
    h, w = partition.shape
    
    # Superpixel contours
    Sc = morphological_gradient(partition)
    Sc[Sc > 0] = 1
    # One-pixel-wide image border Sb
    Sb = add_borders(Sc)

    BR = boundary_recall(Sc, Sb, gt)
    CD = contour_density(Sc, Sb, (h,w))
    MF = average_mismatch_factor(partition)

    return BR, CD, MF
