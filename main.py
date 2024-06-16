import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import skimage
from sklearn.decomposition import PCA


def get_eigenvalues(matrix):
    return np.linalg.eig(matrix)[0]


def get_eigenvectors(matrix):
    return np.linalg.eig(matrix)[1]


any_matrix = np.array([[1, 7, 86], [9, 45, 3], [0, 76, 12]])

eigenvalues = get_eigenvalues(any_matrix)
eigenvectors = get_eigenvectors(any_matrix)

for i in range(len(eigenvalues)):
    A = any_matrix @ eigenvectors[i]
    B = eigenvalues[i] * eigenvectors[i]
    if A.all() == B.all():
        print("Checked")
    else:
        print("Error")

image = imread("woof.jpg")
plt.imshow(image)
plt.show()
print(image.shape)
image2 = image.sum(axis=2)
print(image2.shape)
image_bw = image2/image2.max()
plt.imshow(image_bw)
plt.show()
print(image_bw.max())
print(image_bw.shape)
