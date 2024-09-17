import numpy as np
import matplotlib.pyplot as plt

# Load the laughter and non-laughter features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

# Determine the number of features
num_features = laughter_features.shape[1]

# Loop through each feature and plot the histograms
for feature_index in range(num_features):
    plt.figure(figsize=(10, 6))
    
    plt.hist(laughter_features[:, feature_index], bins=20, alpha=0.7, label='Laughter', color='blue')
    plt.hist(non_laughter_features[:, feature_index], bins=20, alpha=0.7, label='Non-Laughter', color='red')
    
    plt.title(f'Comparison of Feature {feature_index} Between Laughter and Non-Laughter')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Show each plot one at a time
    plt.show()

    # Optionally: Pause to review each plot before moving on to the next
    input("Press Enter to continue to the next feature...")
