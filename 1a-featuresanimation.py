import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the laughter and non-laughter features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

# Determine the number of features
num_features = laughter_features.shape[1]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Function to update the plot for each frame
def update(feature_index):
    ax.clear()  # Clear the previous plot
    ax.hist(laughter_features[:, feature_index], bins=20, alpha=0.7, label='Laughter', color='blue')
    ax.hist(non_laughter_features[:, feature_index], bins=20, alpha=0.7, label='Non-Laughter', color='red')
    
    ax.set_title(f'Comparison of Feature {feature_index} Between Laughter and Non-Laughter')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)

# Create the animation
ani = FuncAnimation(fig, update, frames=num_features, repeat=False)

# Save the animation as a GIF or MP4 (optional)
# ani.save('feature_comparison_animation.gif', writer='imagemagick', fps=2)
# ani.save('feature_comparison_animation.mp4', writer='ffmpeg', fps=2)

# Display the animation
plt.show()
