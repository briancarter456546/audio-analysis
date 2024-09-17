import numpy as np
import matplotlib.pyplot as plt

# Load the laughter and non-laughter features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

# Plot a specific feature across the two classes
feature_index = 0  # Change this index to compare different features

plt.figure(figsize=(10, 6))

plt.hist(laughter_features[:, feature_index], bins=20, alpha=0.7, label='Laughter', color='blue')
plt.hist(non_laughter_features[:, feature_index], bins=20, alpha=0.7, label='Non-Laughter', color='red')

plt.title(f'Comparison of Feature {feature_index} Between Laughter and Non-Laughter')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
