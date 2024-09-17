import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Load the previously saved features
laughter_features = np.load('laughter_features.npy')
non_laughter_features = np.load('non_laughter_features.npy')

print(f"Laughter features shape: {laughter_features.shape}")
print(f"Non-laughter features shape: {non_laughter_features.shape}")

# Function to calculate the score
def calculate_score(segment_features, laughter_ref, speech_ref):
    # Calculate average distances
    laughter_dist = np.mean(euclidean_distances(segment_features.reshape(1, -1), laughter_ref))
    speech_dist = np.mean(euclidean_distances(segment_features.reshape(1, -1), speech_ref))
    
    # Calculate score
    score = (speech_dist - laughter_dist) / (speech_dist + laughter_dist)
    return score, laughter_dist, speech_dist

# Function to classify a new segment
def classify_segment(segment_features, laughter_ref, speech_ref, threshold=0.15):
    score, laughter_dist, speech_dist = calculate_score(segment_features, laughter_ref, speech_ref)
    classification = "Laughter" if score > threshold else "Non-Laughter"
    return classification, score, laughter_dist, speech_dist

# Example usage
# Test with known laughter and non-laughter samples
print("\nTesting with known samples:")
laughter_sample = laughter_features[0]
non_laughter_sample = non_laughter_features[0]

laughter_classification, laughter_score, l_laugh_dist, l_speech_dist = classify_segment(laughter_sample, laughter_features, non_laughter_features)
print(f"Known Laughter Sample - Classification: {laughter_classification}, Score: {laughter_score}")
print(f"  Laughter distance: {l_laugh_dist}, Speech distance: {l_speech_dist}")

non_laughter_classification, non_laughter_score, nl_laugh_dist, nl_speech_dist = classify_segment(non_laughter_sample, laughter_features, non_laughter_features)
print(f"Known Non-Laughter Sample - Classification: {non_laughter_classification}, Score: {non_laughter_score}")
print(f"  Laughter distance: {nl_laugh_dist}, Speech distance: {nl_speech_dist}")

# To classify multiple segments
print("\nClassifying random segments:")
num_segments = 10
new_segments = np.random.rand(num_segments, laughter_features.shape[1])  # Replace with actual new segment features
for i, segment in enumerate(new_segments):
    classification, score, laugh_dist, speech_dist = classify_segment(segment, laughter_features, non_laughter_features)
    print(f"Segment {i+1}: Classification: {classification}, Score: {score}")
    print(f"  Laughter distance: {laugh_dist}, Speech distance: {speech_dist}")

# Print distribution of scores
all_segments = np.vstack([laughter_features, non_laughter_features])
all_scores = np.array([calculate_score(seg, laughter_features, non_laughter_features)[0] for seg in all_segments])
print("\nScore distribution:")
print(f"Min: {np.min(all_scores)}, Max: {np.max(all_scores)}")
print(f"Mean: {np.mean(all_scores)}, Median: {np.median(all_scores)}")
print(f"25th percentile: {np.percentile(all_scores, 25)}, 75th percentile: {np.percentile(all_scores, 75)}")