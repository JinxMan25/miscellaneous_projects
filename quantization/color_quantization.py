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

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))

    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


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

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Origina image')
plt.imshow(china)

plt.figure(2)
plt.clf()
ax = plt.axis('off')
plt.title('Quantized image (64 colors, K-means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()
