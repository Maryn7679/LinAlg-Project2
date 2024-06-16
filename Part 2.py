import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA


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

pca = PCA(n_components=500)
pca.fit_transform(image_bw)

explained_variance = pca.explained_variance_ratio_
components = np.arange(len(explained_variance))
explained_variance_percent = np.array(explained_variance) * 100

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Explained variance by number of components')
ax.set_ylabel('Explained variance (%)')
ax.set_xlabel('Number of PCA components')
ax.plot(components, np.cumsum(explained_variance_percent))

ideal_number_of_components = 0
for variance in np.cumsum(explained_variance_percent):
    ideal_number_of_components += 1
    if variance >= 95:
        plt.axvline(x=ideal_number_of_components, color='black', linestyle=":")
        break
plt.axhline(y=95, color='r', linestyle=':')
plt.show()

pca_ideal = PCA(n_components=ideal_number_of_components)
image_compressed = pca_ideal.fit_transform(image_bw)
image_decompressed = pca_ideal.inverse_transform(image_compressed)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_bw)
axes[0].set_title("Original")
axes[1].imshow(image_decompressed)
axes[1].set_title(f"Components: {ideal_number_of_components}")
plt.show()

fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))

pca_5 = PCA(n_components=5)
image_compressed = pca_5.fit_transform(image_bw)
image_decompressed = pca_5.inverse_transform(image_compressed)
axes2[0, 0].imshow(image_decompressed)
axes2[0, 0].set_title("Components: 5")

pca_15 = PCA(n_components=15)
image_compressed = pca_15.fit_transform(image_bw)
image_decompressed = pca_15.inverse_transform(image_compressed)
axes2[0, 1].imshow(image_decompressed)
axes2[0, 1].set_title("Components: 15")

pca_30 = PCA(n_components=30)
image_compressed = pca_30.fit_transform(image_bw)
image_decompressed = pca_30.inverse_transform(image_compressed)
axes2[0, 2].imshow(image_decompressed)
axes2[0, 2].set_title("Components: 30")

pca_50 = PCA(n_components=50)
image_compressed = pca_50.fit_transform(image_bw)
image_decompressed = pca_50.inverse_transform(image_compressed)
axes2[1, 0].imshow(image_decompressed)
axes2[1, 0].set_title("Components: 50")

pca_70 = PCA(n_components=70)
image_compressed = pca_70.fit_transform(image_bw)
image_decompressed = pca_70.inverse_transform(image_compressed)
axes2[1, 1].imshow(image_decompressed)
axes2[1, 1].set_title("Components: 70")

pca_100 = PCA(n_components=100)
image_compressed = pca_100.fit_transform(image_bw)
image_decompressed = pca_100.inverse_transform(image_compressed)
axes2[1, 2].imshow(image_decompressed)
axes2[1, 2].set_title("Components: 100")

plt.show()
