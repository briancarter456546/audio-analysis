import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the laughter and non-laughter features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

# 1. View the Data
print(f"Laughter features shape: {laughter_features.shape}")
print(f"Non-laughter features shape: {non_laughter_features.shape}")

# View the first few rows of the data
print("Laughter features (first 5 rows):")
print(laughter_features[:5])

print("Non-laughter features (first 5 rows):")
print(non_laughter_features[:5])

# 2. Visualize the Data

# Plot the first feature across the samples
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title("Laughter Features (First Feature)")
plt.hist(laughter_features[:, 0], bins=20, alpha=0.7)
plt.xlabel("Feature Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.title("Non-Laughter Features (First Feature)")
plt.hist(non_laughter_features[:, 0], bins=20, alpha=0.7)
plt.xlabel("Feature Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# 3. Perform PCA to reduce to 2D for visualization
combined_features = np.vstack([laughter_features, non_laughter_features])

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(combined_features)

# Determine how many samples of each type there are
num_laughter = len(laughter_features)
num_non_laughter = len(non_laughter_features)

# Plot the 2D PCA result
plt.figure(figsize=(8, 6))

# Plot laughter samples
plt.scatter(reduced_features[:num_laughter, 0], reduced_features[:num_laughter, 1], 
            color='blue', label='Laughter', alpha=0.5)

# Plot non-laughter samples
plt.scatter(reduced_features[num_laughter:, 0], reduced_features[num_laughter:, 1], 
            color='red', label='Non-Laughter', alpha=0.5)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Laughter vs Non-Laughter Features')
plt.legend()
plt.show()
