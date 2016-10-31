import numpy as np
from scipy.misc import imread, imsave
from sklearn.cluster import KMeans

# Read the image in as a matrix of RGB values
filename = "newfoundland.jpg"
img = imread(filename)
# Get the number of rows, columns, and channels from the image.
# Channels will always be 3 since we have RGB pixels
rows, cols, channels = img.shape
# Now, create a sequential list of all the pixels
pixels = img.reshape((rows * cols, channels))


# Select the number of colors you want in the final picture
# This is also the number of clusters
k = 10
# Create the K-means clusterer
km = KMeans(n_clusters=k)
# Fit it to the pixel data we have. In other words, use this dataset
# and compute the cluster centers.
km.fit(pixels)
# Get the clustering assignments for each of our pixels.
assignments = km.predict(pixels)


# Cluster centroid are 3-tuples of float values, we must convert each
# element to an int to turn them into RGB pixels in an image.
centroids = km.cluster_centers_
colors = [ [int(centroid[0]), int(centroid[1]), int(centroid[2])] for centroid in centroids]
# Map each cluster assignment index to a color.
# Cluster assignments range from 0, 1, 2, ..., k - 1
cluster_colors = {idx : color for idx, color in enumerate(colors)}
# A list of pixels for the new image. Each will be one of k new colors.
new_colors = []
for assignment in assignments:
    color = cluster_colors[assignment]
    new_colors.append(color)


# Turn the list of colors back into a matrix with the same number
# of rows, columns, and channels as the original picture
newimg = np.array(new_colors).reshape((rows, cols, channels))
outname = "newfoundland_{}.jpg".format(k)
# Save this new image and print the image name
imsave(outname, newimg)
print("Saved {}".format(outname))
