import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dtaidistance import dtw_ndim
import pickle

# Load the previously saved features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

print(f"Laughter features shape: {laughter_features.shape}")
print(f"Non-laughter features shape: {non_laughter_features.shape}")

# Normalize the features
scaler = StandardScaler()
laughter_features_normalized = scaler.fit_transform(laughter_features)
non_laughter_features_normalized = scaler.transform(non_laughter_features)

class DTWLaughterClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_references=10):  # Increased number of references
        self.n_references = n_references
        
    def fit(self, X_laughter, X_non_laughter):
        # Randomly select reference sequences
        laughter_indices = np.random.choice(len(X_laughter), self.n_references, replace=False)
        non_laughter_indices = np.random.choice(len(X_non_laughter), self.n_references, replace=False)
        
        self.laughter_references = X_laughter[laughter_indices]
        self.non_laughter_references = X_non_laughter[non_laughter_indices]
        return self
    
    def predict(self, X):
        return (self.decision_function(X) < 0).astype(int)
    
    def decision_function(self, X):
        scores = []
        for x in X:
            x = x.reshape(1, -1)  # Assign reshaped array back to x
            
            laughter_distances = []
            non_laughter_distances = []
            
            for ref in self.laughter_references:
                ref = ref.reshape(1, -1)  # Assign reshaped array back to ref
                distance = dtw_ndim.distance(x, ref, use_c=True)
                laughter_distances.append(distance)
            
            for ref in self.non_laughter_references:
                ref = ref.reshape(1, -1)  # Assign reshaped array back to ref
                distance = dtw_ndim.distance(x, ref, use_c=True)
                non_laughter_distances.append(distance)
            
            avg_laughter_distance = np.mean(laughter_distances)
            avg_non_laughter_distance = np.mean(non_laughter_distances)
            
            score = (avg_non_laughter_distance - avg_laughter_distance) / (avg_non_laughter_distance + avg_laughter_distance)
            scores.append(score)
        return np.array(scores)

# Perform k-fold cross-validation
n_references = 10  # Increased number of references
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
clf = DTWLaughterClassifier(n_references=n_references)

def custom_cross_val_score(clf, X_laughter, X_non_laughter, cv):
    scores = []
    for train_index, test_index in cv.split(X_laughter):
        X_laughter_train, X_laughter_test = X_laughter[train_index], X_laughter[test_index]
        X_non_laughter_train, X_non_laughter_test = X_non_laughter[train_index], X_non_laughter[test_index]
        
        clf.fit(X_laughter_train, X_non_laughter_train)
        
        X_test = np.vstack([X_laughter_test, X_non_laughter_test])
        y_test = np.hstack([np.ones(len(X_laughter_test)), np.zeros(len(X_non_laughter_test))])
        
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.array(scores)

cv_scores = custom_cross_val_score(clf, laughter_features_normalized, non_laughter_features_normalized, kfold)

print("\nCross-validation scores:", cv_scores)
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard deviation of CV accuracy: {np.std(cv_scores):.4f}")

# Now let's train on the full dataset and evaluate on a held-out test set
laughter_train, laughter_test = train_test_split(laughter_features_normalized, test_size=0.2, random_state=42)
non_laughter_train, non_laughter_test = train_test_split(non_laughter_features_normalized, test_size=0.2, random_state=42)

clf.fit(laughter_train, non_laughter_train)

X_test = np.vstack([laughter_test, non_laughter_test])
y_test = np.hstack([np.ones(len(laughter_test)), np.zeros(len(non_laughter_test))])
y_pred = clf.predict(X_test)

# Calculate metrics
cm = confusion_matrix(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("\nClassification Results on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Calculate scores for all samples
all_scores = clf.decision_function(X_test)

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

# Save the trained model
with open('laughter_classifier_dtw_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("\nModel saved as 'laughter_classifier_dtw_model.pkl'")