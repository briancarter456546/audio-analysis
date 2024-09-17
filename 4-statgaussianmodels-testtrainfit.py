import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the previously saved features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

print(f"Laughter features shape: {laughter_features.shape}")
print(f"Non-laughter features shape: {non_laughter_features.shape}")

# Split the data into training and testing sets, maintaining the separation
laughter_train, laughter_test = train_test_split(laughter_features, test_size=0.2, random_state=42)
non_laughter_train, non_laughter_test = train_test_split(non_laughter_features, test_size=0.2, random_state=42)

# Combine train and test sets
X_train = np.vstack([laughter_train, non_laughter_train])
X_test = np.vstack([laughter_test, non_laughter_test])
y_train = np.hstack([np.ones(len(laughter_train)), np.zeros(len(non_laughter_train))])
y_test = np.hstack([np.ones(len(laughter_test)), np.zeros(len(non_laughter_test))])

# Normalize features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Train GMMs
n_components = 5  # You can adjust this
laughter_gmm = GaussianMixture(n_components=n_components, random_state=42)
non_laughter_gmm = GaussianMixture(n_components=n_components, random_state=42)

laughter_gmm.fit(X_train_norm[y_train == 1])
non_laughter_gmm.fit(X_train_norm[y_train == 0])

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

# Classify test set
predicted_labels = []
all_scores = []

for features in X_test:
    classification, score = classify_segment(features, laughter_gmm, non_laughter_gmm)
    predicted_labels.append(classification)
    all_scores.append(score)

predicted_labels = np.array(predicted_labels)
all_scores = np.array(all_scores)

# Calculate metrics
cm = confusion_matrix(y_test, predicted_labels)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predicted_labels, average='binary')
accuracy = accuracy_score(y_test, predicted_labels)

# Print results
print("\nClassification Results on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

# Perform cross-validation
cv_scores = cross_val_score(GaussianMixture(n_components=n_components, random_state=42), X_train_norm, y_train, cv=5)
print("\nCross-validation scores:", cv_scores)
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Plot score distribution
plt.figure(figsize=(10, 6))
plt.hist([score for score, label in zip(all_scores, y_test) if label == 1], bins=30, alpha=0.5, label='Laughter')
plt.hist([score for score, label in zip(all_scores, y_test) if label == 0], bins=30, alpha=0.5, label='Non-Laughter')
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribution of Scores (Test Set)')
plt.legend()
plt.savefig('score_distribution.png')
plt.close()

# Print score distribution statistics
print("\nScore distribution (Test Set):")
print(f"Min: {np.min(all_scores)}, Max: {np.max(all_scores)}")
print(f"Mean: {np.mean(all_scores)}, Median: {np.median(all_scores)}")
print(f"25th percentile: {np.percentile(all_scores, 25)}, 75th percentile: {np.percentile(all_scores, 75)}")

# Save the trained models and scaler
model_data = {
    'laughter_gmm': laughter_gmm,
    'non_laughter_gmm': non_laughter_gmm,
    'scaler': scaler,
    'n_components': n_components
}

with open('laughter_classifier_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel saved as 'laughter_classifier_model.pkl'")
