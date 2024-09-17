import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Load the previously saved features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

print(f"Laughter features shape: {laughter_features.shape}")
print(f"Non-laughter features shape: {non_laughter_features.shape}")

# Normalize features
scaler = StandardScaler()
laughter_features_norm = scaler.fit_transform(laughter_features)
non_laughter_features_norm = scaler.transform(non_laughter_features)

# Train GMMs
n_components = 5  # You can adjust this
laughter_gmm = GaussianMixture(n_components=n_components, random_state=42)
non_laughter_gmm = GaussianMixture(n_components=n_components, random_state=42)

laughter_gmm.fit(laughter_features_norm)
non_laughter_gmm.fit(non_laughter_features_norm)

def calculate_score(segment_features, laughter_gmm, non_laughter_gmm):
    segment_features_norm = scaler.transform(segment_features.reshape(1, -1))
    laughter_log_likelihood = laughter_gmm.score_samples(segment_features_norm)
    non_laughter_log_likelihood = non_laughter_gmm.score_samples(segment_features_norm)
    
    score = laughter_log_likelihood - non_laughter_log_likelihood
    return score[0]

def classify_segment(segment_features, laughter_gmm, non_laughter_gmm, threshold=0):
    score = calculate_score(segment_features, laughter_gmm, non_laughter_gmm)
    classification = 1 if score > threshold else 0  # 1 for Laughter, 0 for Non-Laughter
    return classification, score

# Classify all segments
all_features = np.vstack([laughter_features, non_laughter_features])
true_labels = np.hstack([np.ones(len(laughter_features)), np.zeros(len(non_laughter_features))])

predicted_labels = []
all_scores = []

for features in all_features:
    classification, score = classify_segment(features, laughter_gmm, non_laughter_gmm)
    predicted_labels.append(classification)
    all_scores.append(score)

predicted_labels = np.array(predicted_labels)
all_scores = np.array(all_scores)

# Calculate metrics
cm = confusion_matrix(true_labels, predicted_labels)
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

# Print results
print("\nClassification Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Plot score distribution
plt.figure(figsize=(10, 6))
plt.hist([score for score, label in zip(all_scores, true_labels) if label == 1], bins=30, alpha=0.5, label='Laughter')
plt.hist([score for score, label in zip(all_scores, true_labels) if label == 0], bins=30, alpha=0.5, label='Non-Laughter')
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribution of Scores')
plt.legend()
plt.savefig('score_distribution.png')
plt.close()

# Print score distribution statistics
print("\nScore distribution:")
print(f"Min: {np.min(all_scores)}, Max: {np.max(all_scores)}")
print(f"Mean: {np.mean(all_scores)}, Median: {np.median(all_scores)}")
print(f"25th percentile: {np.percentile(all_scores, 25)}, 75th percentile: {np.percentile(all_scores, 75)}")
