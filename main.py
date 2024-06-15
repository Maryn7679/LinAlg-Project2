import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2


def get_eigenvalues(matrix):
    return np.linalg.eig(matrix)[0]


def get_eigenvectors(matrix):
    return np.linalg.eig(matrix)[1]


any_matrix = np.array([[1, 7, 86], [9, 45, 3], [0, 76, 12]])

for i in range(len(get_eigenvalues(any_matrix))):
    A = any_matrix * get_eigenvectors(any_matrix)[i]
    B = get_eigenvalues(any_matrix)[i] * get_eigenvectors(any_matrix)[i]
    print(A, B)
