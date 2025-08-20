# This script trains a machine learning model to classify music moods
# based on audio features. It now includes a step to balance the
# training data using oversampling to improve accuracy on minority classes.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import librosa
from imblearn.over_sampling import RandomOverSampler # Import the oversampler

def extract_features_from_file(file_path):
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
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def train_mood_model(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio'):
    """
    Loads audio and labels, extracts features, balances the data, and trains a new mood model.
    """
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found at '{labels_file}'")
        return
    if not os.path.isdir(audio_folder):
        print(f"Error: Audio folder not found at '{audio_folder}'")
        print("Please update the 'audio_folder' variable in the script to the correct path.")
        return

    # --- 1. Load Data ---
    print("Loading mood labels...")
    labels_df = pd.read_csv(labels_file)

    all_features = []
    all_labels = []

    print("Starting feature extraction... (This will take a long time)")
    # --- 2. Match Labels to Audio and Extract Features ---
    for index, row in labels_df.iterrows():
        song_id = row['song_id']
        mood_label = row['mood']
        
        file_path = os.path.join(audio_folder, f"{song_id}.mp3")

        if os.path.exists(file_path):
            # print(f"Processing song_id: {song_id}") # Optional: uncomment for verbose output
            features = extract_features_from_file(file_path)
            if features is not None:
                all_features.append(features)
                all_labels.append(mood_label)
        # else:
            # print(f"Warning: Audio file not found for song_id: {song_id}") # Optional: uncomment for verbose output

    if not all_features:
        print("No audio files were successfully processed. Aborting training.")
        return

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\nFeature extraction complete. Found {len(y)} matching tracks.")

    # --- 3. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==============================================================================
    # --- NEW: Balance the Training Data with Oversampling ---
    # ==============================================================================
    print("\nBalancing the training data...")
    print("Original training set distribution:")
    print(pd.Series(y_train).value_counts())
    
    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    print("\nNew, balanced training set distribution:")
    print(pd.Series(y_train_resampled).value_counts())
    # ==============================================================================

    # --- 4. Scale the Features ---
    mood_scaler = StandardScaler()
    # Fit the scaler ONLY on the original training data, then transform both sets
    X_train_scaled = mood_scaler.fit_transform(X_train_resampled)
    X_test_scaled = mood_scaler.transform(X_test)
    
    print("\nTraining the new, balanced mood model...")
    # Train on the new, resampled data
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_train_scaled, y_train_resampled)
    print("Training complete!")

    # --- 5. Evaluate and Save ---
    # Evaluate on the original, untouched test data
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nMood Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    model_filename = 'mood_classifier_model.joblib'
    scaler_filename = 'mood_feature_scaler.joblib'
    
    print(f"\nSaving new model to '{model_filename}'")
    joblib.dump(classifier, model_filename)
    
    print(f"Saving new scaler to '{scaler_filename}'")
    joblib.dump(mood_scaler, scaler_filename)
    
    print("\nDone! Your improved custom mood model is now ready.")

if __name__ == "__main__":
    # !!! IMPORTANT !!!
    # You MUST update this path to point to the folder where you
    # have downloaded the audio files that correspond to the CSVs.
    path_to_audio_files = '/home/j/AI-Genre-Tagger/' # Example path
    
    train_mood_model(audio_folder=path_to_audio_files)
