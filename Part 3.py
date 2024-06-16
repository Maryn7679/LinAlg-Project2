import numpy as np


def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector, message_vector


def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    message_vector = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector)
    message = ""
    for char in np.real(np.around(message_vector)).astype(np.int64):
        message += chr(char)
    return message


my_message = "Hello, World!"
my_key_matrix = np.random.randint(0, 256, (len(my_message), len(my_message)))
encrypted, mess = encrypt_message(my_message, my_key_matrix)
print(decrypt_message(encrypted, my_key_matrix))
