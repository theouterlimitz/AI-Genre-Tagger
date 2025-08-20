# This script scans a directory of audio files, analyzes them using
# trained ML models to predict both genre and mood, and saves all
# results into a comprehensive CSV file.

import os
import csv
import librosa
import numpy as np
import mutagen
from mutagen.easyid3 import EasyID3
import joblib

# --- Global variables for our trained models and scalers ---
GENRE_CLASSIFIER = None
GENRE_SCALER = None
MOOD_CLASSIFIER = None
MOOD_SCALER = None

def load_models():
    """
    Loads both the genre and mood machine learning models from disk.
    """
    global GENRE_CLASSIFIER, GENRE_SCALER, MOOD_CLASSIFIER, MOOD_SCALER
    
    # --- Load Genre Model ---
    genre_model_path = 'genre_classifier_model.joblib'
    genre_scaler_path = 'feature_scaler.joblib'
    if os.path.exists(genre_model_path) and os.path.exists(genre_scaler_path):
        print("Loading Genre model...")
        GENRE_CLASSIFIER = joblib.load(genre_model_path)
        GENRE_SCALER = joblib.load(genre_scaler_path)
        print("Genre model loaded successfully.")
    else:
        print("!!! WARNING: Genre model files not found. Genre prediction will be disabled.")

    # --- Load Mood Model ---
    mood_model_path = 'mood_classifier_model.joblib'
    mood_scaler_path = 'mood_feature_scaler.joblib'
    if os.path.exists(mood_model_path) and os.path.exists(mood_scaler_path):
        print("Loading Mood model...")
        MOOD_CLASSIFIER = joblib.load(mood_model_path)
        MOOD_SCALER = joblib.load(mood_scaler_path)
        print("Mood model loaded successfully.")
    else:
        print("!!! WARNING: Mood model files not found. Mood prediction will be disabled.")

    return True

def get_audio_features(file_path):
    """
    Extracts a universal set of audio features from a single audio file.
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
    Uses the loaded genre model to predict the genre.
    """
    if GENRE_CLASSIFIER is None:
        return "Not Available"
    
    features_scaled = GENRE_SCALER.transform(features)
    prediction = GENRE_CLASSIFIER.predict(features_scaled)
    return prediction[0]

def predict_mood(features):
    """
    Uses the loaded mood model to predict the mood.
    """
    if MOOD_CLASSIFIER is None:
        return "Not Available"
        
    features_scaled = MOOD_SCALER.transform(features)
    prediction = MOOD_CLASSIFIER.predict(features_scaled)
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

            song_data = {'filename': filename, 'title': 'Unknown', 'artist': 'Unknown', 'genre': 'Unknown', 'mood': 'Unknown'}
            
            # --- Read Existing Tags ---
            try:
                audio = mutagen.File(file_path, easy=True)
                if audio:
                    song_data['title'] = audio.get('title', [os.path.splitext(filename)[0]])[0]
                    song_data['artist'] = audio.get('artist', ['Unknown'])[0]
                    song_data['genre'] = audio.get('genre', ['Unknown'])[0]
            except Exception:
                print(f"    -> No readable tags found for {filename}.")

            # --- Analyze and Predict ---
            print("    -> Analyzing audio features...")
            features = get_audio_features(file_path)
            
            if features is not None:
                # Predict Genre if it's missing
                if song_data['genre'] == 'Unknown':
                    predicted_genre = predict_genre(features)
                    song_data['genre'] = predicted_genre
                    print(f"    -> Predicted Genre: {predicted_genre}")
                else:
                    print(f"    -> Found existing genre: {song_data['genre']}")
                
                # Always predict mood
                predicted_mood = predict_mood(features)
                song_data['mood'] = predicted_mood
                print(f"    -> Predicted Mood: {predicted_mood}")

            music_library.append(song_data)

    output_csv_path = os.path.join(folder_path, 'my_music_library_classified.csv')
    print(f"\n...Scan complete. Saving library data to {output_csv_path}")
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Add 'mood' to the output file
            fieldnames = ['filename', 'title', 'artist', 'genre', 'mood']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(music_library)
        print("Successfully created CSV file!")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")

if __name__ == "__main__":
    if load_models():
        path = input("Enter the path to your music folder: ")
        process_music_folder(path)
