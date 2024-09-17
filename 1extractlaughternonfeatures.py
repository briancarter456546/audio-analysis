import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Function to extract features using librosa
def extract_features(file_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Flatten and concatenate the features to ensure consistent shape
    features = np.concatenate([
        np.mean(mfcc, axis=1), 
        np.mean(spectral_contrast, axis=1), 
        np.mean(chroma, axis=1)
    ])
    
    return features


# Function to process a folder of audio files and extract features
def process_folder(folder_path):
    features = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav') or file_name.endswith('.mp3'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            feature = extract_features(file_path)
            features.append(feature)
    return np.array(features)

# Select laughter and non-laughter folders using tkinter
def select_folders():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    print("Select folder containing laughter samples:")
    laughter_folder = filedialog.askdirectory()

    print("Select folder containing non-laughter samples:")
    non_laughter_folder = filedialog.askdirectory()

    return laughter_folder, non_laughter_folder

# Main function to run the process
def main():
    laughter_folder, non_laughter_folder = select_folders()

    print("Extracting features from laughter samples...")
    laughter_features = process_folder(laughter_folder)
    print(f"Extracted {laughter_features.shape[0]} laughter features.")

    print("Extracting features from non-laughter samples...")
    non_laughter_features = process_folder(non_laughter_folder)
    print(f"Extracted {non_laughter_features.shape[0]} non-laughter features.")

    # You can save these features for later use
    np.save('laughter_features.npy', laughter_features)
    np.save('non_laughter_features.npy', non_laughter_features)
    print("Features saved as 'laughter_features.npy' and 'non_laughter_features.npy'.")

if __name__ == "__main__":
    main()
