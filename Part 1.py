import numpy as np


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
        