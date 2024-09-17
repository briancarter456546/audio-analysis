import os
import numpy as np
import matplotlib.pyplot as plt

# Create a directory to store the images
output_dir = 'histogram_images'
os.makedirs(output_dir, exist_ok=True)

# Load the laughter and non-laughter features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

# Determine the number of features
num_features = laughter_features.shape[1]

# Loop through each feature and save the histograms as images
for feature_index in range(num_features):
    plt.figure(figsize=(10, 6))
    
    plt.hist(laughter_features[:, feature_index], bins=20, alpha=0.7, label='Laughter', color='blue')
    plt.hist(non_laughter_features[:, feature_index], bins=20, alpha=0.7, label='Non-Laughter', color='red')
    
    plt.title(f'Comparison of Feature {feature_index} Between Laughter and Non-Laughter')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image file
    output_path = os.path.join(output_dir, f'feature_{feature_index}.png')
    plt.savefig(output_path)
    plt.close()

    print(f'Saved: {output_path}')

import imageio
import os

# Directory where images are saved
output_dir = 'histogram_images'
images = []

# Load the images and create the GIF
for feature_index in range(num_features):
    filename = os.path.join(output_dir, f'feature_{feature_index}.png')
    images.append(imageio.imread(filename))

# Save the images as an animated GIF
imageio.mimsave('histogram_animation.gif', images, fps=2)
