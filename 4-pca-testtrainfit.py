import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the previously saved features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

print(f"Laughter features shape: {laughter_features.shape}")
print(f"Non-laughter features shape: {non_laughter_features.shape}")

# Define a custom classifier class for the PCA-based model
class PCALaughterClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        
    def fit(self, X_laughter, X_non_laughter):
        X = np.vstack([X_laughter, X_non_laughter])
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.laughter_mean = np.mean(X_pca[:len(X_laughter)], axis=0)
        self.non_laughter_mean = np.mean(X_pca[len(X_laughter):], axis=0)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        scores = np.dot(X_pca, self.laughter_mean - self.non_laughter_mean)
        return (scores > 0).astype(int)

# Perform k-fold cross-validation
n_components = 5  # You can adjust this
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
clf = PCALaughterClassifier(n_components=n_components)

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

cv_scores = custom_cross_val_score(clf, laughter_features, non_laughter_features, kfold)

print("\nCross-validation scores:", cv_scores)
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard deviation of CV accuracy: {np.std(cv_scores):.4f}")

# Now let's train on the full dataset and evaluate on a held-out test set
laughter_train, laughter_test = train_test_split(laughter_features, test_size=0.2, random_state=42)
non_laughter_train, non_laughter_test = train_test_split(non_laughter_features, test_size=0.2, random_state=42)

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
all_scores = np.dot(clf.pca.transform(clf.scaler.transform(X_test)), 
                    clf.laughter_mean - clf.non_laughter_mean)

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
with open('laughter_classifier_pca_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("\nModel saved as 'laughter_classifier_pca_model.pkl'")
