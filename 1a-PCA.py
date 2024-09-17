from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the laughter and non-laughter features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

# Combine the features
combined_features = np.vstack([laughter_features, non_laughter_features])

# Apply PCA
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
plt.grid(True)
plt.show()
