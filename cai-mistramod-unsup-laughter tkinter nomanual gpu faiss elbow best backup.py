#!/usr/bin/env python
# coding: utf-8

print("Starting file selection...")

# Import necessary libraries and define helper functions for selecting files and loading audio.
import librosa
import numpy as np
import torch
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import filedialog
import faiss  # Import faiss for GPU-accelerated KMeans
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

# Global variable to store the file path
selected_file_path = None

# Function to select file using tkinter
def select_file():
    global selected_file_path
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    selected_file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3")]
    )
    if selected_file_path:
        print(f"Selected file: {selected_file_path}")
    else:
        print("No file selected")

# Example usage
select_file()

print("Success")

# In[2]:

print("Loading audio file...")

# Select audio file and load audio with lower sample rate for faster processing.
# Check if a file has been selected
if not selected_file_path:
    raise ValueError("No file selected")

# Load audio file with a lower sample rate for faster processing
y, sr = librosa.load(selected_file_path, sr=16000)
print("Success")
print(f"Audio shape: {y.shape}, sample rate: {sr}")

# In[3]:

print("Segmenting audio...")

# Segment audio into 3-second chunks and pad segments to ensure uniform length.
# Initial segmentation and feature extraction
segment_length = 3  # seconds
hop_length = sr * segment_length
segments = [y[i:i + hop_length] for i in range(0, len(y), hop_length)]

# Pad segments to ensure uniform length
max_length = max(len(seg) for seg in segments)
padded_segments = [np.pad(seg, (0, max_length - len(seg)), 'constant') for seg in segments]

print(f"Number of segments: {len(segments)}, padded segment shape: {padded_segments[0].shape}")

# In[4]:

print("Extracting features...")

# Extract features from segments using MFCCs or a CNN.

# Extract features using MFCCs
features = [librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13).flatten() for seg in padded_segments]

# Convert the list of features to a NumPy array
features_array = np.array(features)

print(f"Features shape: {features_array.shape}")

# In[5]:

print("Determining optimal number of clusters using faiss KMeans...")

# Determine optimal number of clusters using faiss KMeans
scores = []
k_values = range(2, 11)
for k in k_values:
    kmeans = faiss.Kmeans(d=features_array.shape[1], k=k, gpu=True)
    kmeans.train(features_array)
    labels = kmeans.index.search(features_array, 1)[1].squeeze()
    score = silhouette_score(features_array, labels)
    scores.append(score)

# Find the elbow point manually
differences = np.diff(scores)
optimal_k = k_values[np.argmin(differences) + 1]

# Plot scores and elbow point
plt.figure(figsize=(10, 5))
plt.plot(k_values, scores, 'bo-')
plt.plot(optimal_k, scores[optimal_k-2], 'ro', markersize=12, label='Elbow')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')
plt.title('Elbow Method for Optimal k')
plt.legend()
plt.show()

kmeans = faiss.Kmeans(d=features_array.shape[1], k=optimal_k, gpu=True)
kmeans.train(features_array)
labels = kmeans.index.search(features_array, 1)[1].squeeze()

print(f"Optimal number of clusters: {optimal_k}")
print(f"Silhouette scores: {scores}")

# In[6]:

print("Labeling clusters...")

# Use cluster labels directly
labeled_segments = [(seg, label) for seg, label in zip(padded_segments, labels)]

# In[7]:

# Skip the transcription part for now

# In[8]:

print("Creating timeline...")

# Use hierarchical clustering to segment audio into events and create timeline.
from sklearn.cluster import AgglomerativeClustering

# Use hierarchical clustering to segment audio into events
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
event_labels = clustering.fit_predict(features_array)

# Create timeline based on events
timeline = []
current_event = None
for i, (seg, event_label, cluster_label) in enumerate(zip(padded_segments, event_labels, labels)):
    if event_label != current_event:
        # Start new event
        current_event = event_label
        start_time = i * segment_length
        end_time = start_time + segment_length
        text = ''
        reaction = event_label  # Use event label as reaction
        cluster = int(cluster_label)  # Add cluster label
    else:
        # Continue current event
        end_time += segment_length

    if i == len(padded_segments) - 1 or event_labels[i + 1] != current_event:
        # End current event
        timeline.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': text.strip(),
            'reaction': reaction,
            'cluster': cluster
        })

timeline_length = timeline[-1]['end_time'] - timeline[0]['start_time']
print(f"Number of events in timeline: {len(timeline)}, total timeline length: {timeline_length}")

# In[9]:

print("Writing timeline to file...")

# Get the base name of the source audio file
audio_file_name = os.path.basename(selected_file_path)
# Take the first 10 characters of the audio file name
audio_file_prefix = audio_file_name[:10]

# Get current date and time
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the new filename
csv_filename = f"timeline_{audio_file_prefix}_{current_datetime}.csv"

# Write timeline to CSV file
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['start_time', 'end_time', 'text', 'reaction', 'cluster']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for event in timeline:
        writer.writerow(event)

print(f"Timeline written to {csv_filename}")

# Print timeline
print("Timeline:")
for i, event in enumerate(timeline):
    print(f"Event {i+1}: Start: {event['start_time']}, End: {event['end_time']}, Text: {event['text']}, Reaction: {event['reaction']}, Cluster: {event['cluster']}")
    if i >= 4:
        break

print("Done.")
