# This script scans a directory of audio files, reads their metadata,
# analyzes them using a trained ML model to predict the genre,
# and saves all results into a CSV file.
# NOTE: This version does NOT write tags back to the original audio files.

import os
import csv
import librosa
import numpy as np
import mutagen
from mutagen.easyid3 import EasyID3
import joblib

# --- Global variables to hold our trained model and scaler ---
CLASSIFIER_MODEL = None
FEATURE_SCALER = None

def load_ml_model(model_path='genre_classifier_model.joblib', scaler_path='feature_scaler.joblib'):
    """
    Loads the trained machine learning model and feature scaler from disk.
    """
    global CLASSIFIER_MODEL, FEATURE_SCALER
    if not all(os.path.exists(p) for p in [model_path, scaler_path]):
        print("="*50)
        print("!!! WARNING: Model files not found. !!!")
        print(f"Cannot find '{model_path}' or '{scaler_path}'.")
        print("Genre prediction will be disabled.")
        print("Please run the 'train_custom_model.py' script first.")
        print("="*50)
        return False
    
    print("Loading ML model and feature scaler...")
    CLASSIFIER_MODEL = joblib.load(model_path)
    FEATURE_SCALER = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
    return True

def get_audio_features(file_path):
    """
    Extracts a set of audio features from a single audio file.
    This is the "fingerprint" we create for each song.
    """
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        features = np.array([
            np.mean(chroma_stft), np.var(chroma_stft),
            np.mean(rms), np.var(rms),
            np.mean(spec_cent), np.var(spec_cent),
            np.mean(spec_bw), np.var(spec_bw),
            np.mean(rolloff), np.var(rolloff),
            np.mean(zcr), np.var(zcr),
            *np.mean(mfccs, axis=1), *np.var(mfccs, axis=1)
        ])
        
        return features.reshape(1, -1)

    except Exception as e:
        print(f"    -> Could not extract features from {os.path.basename(file_path)}. Error: {e}")
        return None

def predict_genre(features):
    """
    Uses the loaded model to predict the genre for a set of audio features.
    """
    if CLASSIFIER_MODEL is None or FEATURE_SCALER is None:
        return "ML Model not loaded"
    
    features_scaled = FEATURE_SCALER.transform(features)
    prediction = CLASSIFIER_MODEL.predict(features_scaled)
    return prediction[0]

def process_music_folder(folder_path):
    """
    Main function to scan folder, analyze files, and save results to a CSV.
    """
    music_library = []
    print(f"\nScanning '{folder_path}' for audio files...\n")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mp3', '.wav')):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            song_data = {'filename': filename, 'title': 'Unknown', 'artist': 'Unknown', 'genre': 'Unknown'}
            
            try:
                audio = mutagen.File(file_path, easy=True)
                if audio:
                    song_data['title'] = audio.get('title', [os.path.splitext(filename)[0]])[0]
                    song_data['artist'] = audio.get('artist', ['Unknown'])[0]
                    song_data['genre'] = audio.get('genre', ['Unknown'])[0]
            except Exception:
                print(f"    -> No readable tags found for {filename}.")

            if song_data['genre'] == 'Unknown':
                print("    -> Genre not found in tags. Analyzing with ML model...")
                features = get_audio_features(file_path)
                if features is not None:
                    predicted_genre = predict_genre(features)
                    song_data['genre'] = predicted_genre
                    print(f"    -> Predicted Genre: {predicted_genre}")
            else:
                print(f"    -> Found existing genre: {song_data['genre']}")

            music_library.append(song_data)

    output_csv_path = os.path.join(folder_path, 'my_music_library_classified.csv')
    print(f"\n...Scan complete. Saving library data to {output_csv_path}")
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'title', 'artist', 'genre']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(music_library)
        print("Successfully created CSV file!")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    if load_ml_model():
        path = input("Enter the path to your music folder: ")
        process_music_folder(path)
