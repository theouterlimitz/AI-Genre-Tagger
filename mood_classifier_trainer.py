# This script trains a Neural Network to classify music moods.
# It uses data augmentation (adding noise, time stretching, pitch shifting)
# to create a more robust and balanced training set.

import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import librosa
import tensorflow as tf

def augment_audio(y, sr):
    """
    Applies a random augmentation to the audio data.
    """
    # Choose a random augmentation
    aug_type = random.choice(['noise', 'stretch', 'pitch'])

    if aug_type == 'noise':
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    elif aug_type == 'stretch':
        rate = random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y=y, rate=rate)
    elif aug_type == 'pitch':
        n_steps = random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
        
    return y

def extract_features_from_data(y, sr):
    """
    Extracts a set of audio features from audio data (y).
    """
    try:
        features_to_extract = [
            librosa.feature.chroma_stft, librosa.feature.rms,
            librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth,
            librosa.feature.spectral_rolloff, librosa.feature.zero_crossing_rate
        ]
        
        all_features = []
        for func in features_to_extract:
            if 'sr' in func.__code__.co_varnames:
                feature = func(y=y, sr=sr)
            else:
                feature = func(y=y)
            all_features.extend([np.mean(feature), np.var(feature)])
            
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        all_features.extend(np.mean(mfccs, axis=1))
        all_features.extend(np.var(mfccs, axis=1))
        
        return np.array(all_features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def train_augmented_mood_model(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio'):
    """
    Loads data, creates an augmented and balanced dataset, and trains a neural network.
    """
    if not os.path.exists(labels_file) or not os.path.isdir(audio_folder):
        print("Error: Ensure labels file and audio folder exist and paths are correct.")
        return

    print("Loading mood labels...")
    labels_df = pd.read_csv(labels_file)

    # --- 1. Load Original Audio and Prepare Data Structures ---
    print("Loading original audio data...")
    audio_data = {}
    for index, row in labels_df.iterrows():
        song_id = row['song_id']
        file_path = os.path.join(audio_folder, f"{song_id}.mp3")
        if os.path.exists(file_path):
            try:
                y, sr = librosa.load(file_path, mono=True, duration=30)
                audio_data[song_id] = {'y': y, 'sr': sr, 'mood': row['mood']}
            except Exception as e:
                print(f"Could not load {file_path}: {e}")

    # --- 2. Data Augmentation and Feature Extraction ---
    print("\nStarting data augmentation and feature extraction...")
    all_features = []
    all_labels = []
    
    mood_counts = labels_df['mood'].value_counts()
    target_count = mood_counts.max() # Get the count of the majority class
    print(f"Balancing all classes to match majority class count: {target_count}")

    for mood, count in mood_counts.items():
        print(f"Processing mood: {mood}")
        
        # Get all song_ids for the current mood
        mood_song_ids = labels_df[labels_df['mood'] == mood]['song_id'].tolist()
        
        # Add original samples
        for song_id in mood_song_ids:
            if song_id in audio_data:
                features = extract_features_from_data(audio_data[song_id]['y'], audio_data[song_id]['sr'])
                if features is not None:
                    all_features.append(features)
                    all_labels.append(mood)
        
        # Add augmented samples until we reach the target count
        num_to_augment = target_count - count
        for i in range(num_to_augment):
            # Choose a random song from this mood to augment
            random_song_id = random.choice(mood_song_ids)
            if random_song_id in audio_data:
                original_audio = audio_data[random_song_id]
                augmented_y = augment_audio(original_audio['y'], original_audio['sr'])
                features = extract_features_from_data(augmented_y, original_audio['sr'])
                if features is not None:
                    all_features.append(features)
                    all_labels.append(mood)

    X = np.array(all_features)
    y_text = np.array(all_labels)

    print("\nNew, augmented and balanced dataset distribution:")
    print(pd.Series(y_text).value_counts())

    # --- 3. Encode, Split, and Scale ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Build and Train the Neural Network ---
    print("\nBuilding and training the Neural Network model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=2)

    # --- 5. Evaluate and Save ---
    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Augmented Mood Model Accuracy: {accuracy * 100:.2f}%")
    
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model.save('mood_nn_augmented_model.keras')
    joblib.dump(scaler, 'mood_nn_augmented_scaler.joblib')
    joblib.dump(label_encoder, 'mood_label_encoder.joblib') # Can overwrite, it's the same
    
    print("\nDone! Your augmented Neural Network mood model is ready.")

if __name__ == "__main__":
    path_to_audio_files = '/home/j/AI-Genre-Tagger/' # Example path
    train_augmented_mood_model(audio_folder=path_to_audio_files)