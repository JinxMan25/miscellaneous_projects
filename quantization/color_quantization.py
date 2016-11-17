# Color quantization using using K-means
# License: BSD 3 clause

print (__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import pdb

n_colors = 64

#Load photo
photo = load_sample_image("china.jpg")


# Convert image array to floats instead of 8 bit integer coding. 
# Divide by 255 so that plt.imshow works well on floating points
# Normalizes data (puts in [0-1] range)
china = np.array(photo, dtype=np.float64) / 255

w, h, d = original_shape = tuple(china.shape)

image_array = np.reshape(china, (w*h, d))

image_array_sample = shuffle(image_array, random_state=0)[:1000]

kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

labels = kmeans.predict(image_array)

codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)

