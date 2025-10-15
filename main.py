import fonctions as fct

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

def main():
    fct.viewimage(fct.derive('Lake.png'), gray=True )