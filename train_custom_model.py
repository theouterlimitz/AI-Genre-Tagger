# This script trains a new genre classification model based on a folder
# of audio files that you provide. It uses the subfolder names as genre labels.

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import librosa

def extract_features_from_file(file_path):
    """
    Extracts a set of audio features from a single audio file.
    This is the "fingerprint" we create for each song.
    """
    try:
        # Load 30 seconds of the audio file
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        # --- Extract a robust set of features ---
        # We'll calculate the mean and variance for each feature type.
        
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Concatenate all the features into a single feature vector
        features = np.array([
            np.mean(chroma_stft), np.var(chroma_stft),
            np.mean(rms), np.var(rms),
            np.mean(spec_cent), np.var(spec_cent),
            np.mean(spec_bw), np.var(spec_bw),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zcr), np.var(zcr),
            *np.mean(mfccs, axis=1), *np.var(mfccs, axis=1) # Unpack means and variances of all 20 MFCCs
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def train_custom_model(data_folder):
    """
    Loads audio from subfolders, extracts features, and trains a new model.
    """
    if not os.path.isdir(data_folder):
        print(f"Error: '{data_folder}' is not a valid directory.")
        return

    all_features = []
    all_labels = []

    # --- 1. Load Data and Extract Features ---
    print("Starting data preparation...")
    # os.walk goes through all subdirectories and files
    for root, dirs, files in os.walk(data_folder):
        for genre_folder in dirs:
            print(f"Processing genre: {genre_folder}")
            genre_path = os.path.join(root, genre_folder)
            for filename in os.listdir(genre_path):
                if filename.lower().endswith(('.mp3', '.wav')):
                    file_path = os.path.join(genre_path, filename)
                    
                    # Extract features and add to our lists
                    features = extract_features_from_file(file_path)
                    if features is not None:
                        all_features.append(features)
                        all_labels.append(genre_folder)
    
    if not all_features:
        print("No audio files were successfully processed. Aborting training.")
        return

    # Convert lists to numpy arrays for scikit-learn
    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\nData preparation complete. Found {len(y)} tracks across {len(np.unique(y))} genres.")

    # --- 2. Split, Scale, and Train ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining the new custom model... (This may take a few minutes)")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_train_scaled, y_train)
    print("Training complete!")

    # --- 3. Evaluate and Save ---
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy on your custom data: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the new model and scaler, overwriting the old ones
    model_filename = 'genre_classifier_model.joblib'
    scaler_filename = 'feature_scaler.joblib'
    
    print(f"\nSaving new model to '{model_filename}'")
    joblib.dump(classifier, model_filename)
    
    print(f"Saving new scaler to '{scaler_filename}'")
    joblib.dump(scaler, scaler_filename)
    
    print("\nDone! Your custom model is now ready to be used by the organizer script.")

if __name__ == "__main__":
    # The path to the folder containing your genre subfolders
    path_to_training_data = 'training_data'
    train_custom_model(path_to_training_data)