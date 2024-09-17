import os
import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pickle
import pandas as pd
from datetime import datetime
from classifier import PCALaughterClassifier

# Function to extract features using librosa
def extract_features(y, sr, n_mfcc=13):
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

# Function to load all pickle models from the models directory
def load_models(models_dir='models'):
    model_path = f"{models_dir}/laughter_classifier_model_gaussian.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to classify a feature vector with all loaded models
def classify_segment(features, models):
    classifications = {}
    confidences = {}
    for model_name, model in models.items():
        try:
            # Assuming models have a predict_proba method
            proba = model.predict_proba([features])[0]
            confidence = np.max(proba) * 100  # Confidence percentage
            prediction = model.predict([features])[0]
            
            classifications[model_name] = prediction
            confidences[model_name] = confidence
        except AttributeError:
            # If the model doesn't have predict_proba, fallback to predict
            prediction = model.predict([features])[0]
            classifications[model_name] = prediction
            confidences[model_name] = None  # Confidence not available
            print(f"Model '{model_name}' does not support 'predict_proba'. Confidence set to None.")
    return classifications, confidences

# Function to split audio into 2-second segments
def split_audio(y, sr, segment_length=2):
    total_duration = librosa.get_duration(y=y, sr=sr)
    segments = []
    for start in np.arange(0, total_duration, segment_length):
        end = start + segment_length
        if end > total_duration:
            end = total_duration
        segments.append((start, end))
    return segments

# Function to select an audio file using tkinter
def select_audio_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3"), ("All Files", "*.*"))
    )
    return file_path

# Function to process the audio file
def process_audio(file_path, models, segment_length=2):
    y, sr = librosa.load(file_path, sr=16000)
    segments = split_audio(y, sr, segment_length)
    results = []
    
    for start, end in segments:
        # Extract the segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_y = y[start_sample:end_sample]
        
        # Extract features
        features = extract_features(segment_y, sr)
        
        # Classify the segment
        classifications, confidences = classify_segment(features, models)
        
        # Prepare the result dictionary
        result = {
            'start_time': start,
            'end_time': end
        }
        
        for model_name in models.keys():
            classification_key = f'classification_{model_name}'
            confidence_key = f'confidence_{model_name}'
            result[classification_key] = classifications.get(model_name, None)
            result[confidence_key] = confidences.get(model_name, None)
        
        results.append(result)
    
    return results

# Main function to run the classification process
def main():
    audio_file = select_audio_file()
    if not audio_file:
        print("No file selected.")
        return
    
    # Define the models directory
    models_dir = 'models'
    
    try:
        # Load the models
        print("Loading models...")
        models = load_models(models_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading models: {e}")
        return
    
    # Process the audio file
    print(f"Processing '{os.path.basename(audio_file)}'...")
    results = process_audio(audio_file, models, segment_length=2)
    
    # Create a DataFrame
    df = pd.DataFrame(results)
    
    # Generate CSV file name
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"{base_name}_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(csv_name, index=False)
    print(f"Classification results saved to '{csv_name}'.")

if __name__ == "__main__":
    main()