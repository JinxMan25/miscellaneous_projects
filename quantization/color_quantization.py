# Color quantization using using K-means
# License: BSD 3 clause

print (__doc__)
import numpy as np
import matplotlib.pyplot as plt
from skylearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import import shuffle
from time import time

n_colors = 64

#Load photo
photo = load_sample_image("sample.jpg")

# Convert image array to floats instead of 8 bit integer coding. 
# Divide by 255 so that plt.imshow works well on floating points
# Normalizes data (puts in [0-1] range)
photo = np.array(photo, dtype=np.float64) / 255
