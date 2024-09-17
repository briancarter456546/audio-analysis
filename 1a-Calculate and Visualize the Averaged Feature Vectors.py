import numpy as np
import matplotlib.pyplot as plt

# Load the laughter and non-laughter features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

# Calculate the mean feature vector for each group
mean_laughter_features = np.mean(laughter_features, axis=0)
mean_non_laughter_features = np.mean(non_laughter_features, axis=0)

# Plot the mean feature vectors
plt.figure(figsize=(12, 6))

plt.plot(mean_laughter_features, label='Mean Laughter Features', marker='o', linestyle='-', color='blue')
plt.plot(mean_non_laughter_features, label='Mean Non-Laughter Features', marker='o', linestyle='-', color='red')

plt.title('Mean Feature Vectors for Laughter and Non-Laughter')
plt.xlabel('Feature Index')
plt.ylabel('Mean Feature Value')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the standard deviation for each feature across all samples in each group
std_laughter_features = np.std(laughter_features, axis=0)
std_non_laughter_features = np.std(non_laughter_features, axis=0)

# Plot the standard deviation of feature vectors
plt.figure(figsize=(12, 6))

plt.plot(std_laughter_features, label='Laughter Feature Std Dev', marker='o', linestyle='-', color='blue')
plt.plot(std_non_laughter_features, label='Non-Laughter Feature Std Dev', marker='o', linestyle='-', color='red')

plt.title('Standard Deviation of Features for Laughter and Non-Laughter')
plt.xlabel('Feature Index')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()
