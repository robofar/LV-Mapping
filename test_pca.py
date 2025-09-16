import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Artificial centroids moving diagonally with slight curvature
np.random.seed(0)
num_points = 20
x_world = np.linspace(0, 10, num_points)
y_world = 0.5 * x_world + 0.05 * x_world**2 + np.random.normal(0, 0.1, num_points)

centroids_world = np.vstack((x_world, y_world)).T

# ---- Fit polynomial in world frame ----
poly_world = np.polyfit(x_world, y_world, deg=2)

# ---- PCA ----
pca = PCA(n_components=2)
centroids_pca = pca.fit_transform(centroids_world)

x_pca = centroids_pca[:, 0]
y_pca = centroids_pca[:, 1]

# Fit polynomial in PCA frame
poly_pca = np.polyfit(x_pca, y_pca, deg=2)

# ---- Plot in world frame ----
plt.figure(figsize=(12, 5))

# Plot world frame fit
plt.subplot(1, 2, 1)
plt.scatter(x_world, y_world, label='Centroids (world frame)', color='blue')
x_fit_world = np.linspace(min(x_world), max(x_world), 100)
y_fit_world = np.polyval(poly_world, x_fit_world)
plt.plot(x_fit_world, y_fit_world, label='Fitted poly (world frame)', color='green')
plt.title("World Frame")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis("equal")

# ---- Plot in PCA frame ----
plt.subplot(1, 2, 2)
plt.scatter(x_pca, y_pca, label='Centroids (PCA frame)', color='blue')
x_fit_pca = np.linspace(min(x_pca), max(x_pca), 100)
y_fit_pca = np.polyval(poly_pca, x_fit_pca)
plt.plot(x_fit_pca, y_fit_pca, label='Fitted poly (PCA frame)', color='green')
plt.title("PCA Frame")
plt.xlabel("PCA-X")
plt.ylabel("PCA-Y")
plt.legend()
plt.axis("equal")

plt.tight_layout()
plt.show()
