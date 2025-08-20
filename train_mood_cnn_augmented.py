# This script trains a Convolutional Neural Network (CNN) on a balanced,
# augmented dataset of spectrograms to create our most robust mood classifier.

import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# --- Configuration ---
IMG_HEIGHT = 128
IMG_WIDTH = 128

def augment_audio(y, sr):
    """
    Applies a random augmentation to the audio data.
    """
    aug_type = random.choice(['noise', 'stretch', 'pitch', 'none'])

    if aug_type == 'noise':
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    elif aug_type == 'stretch':
        rate = random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y=y, rate=rate)
    elif aug_type == 'pitch':
        n_steps = random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    # The 'none' case returns the original audio
    return y

def create_spectrogram_from_data(y, sr):
    """
    Creates a Mel Spectrogram from audio data (y) and returns it as an image array.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize the spectrogram to be between 0 and 1
    S_db_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))
    
    # Resize to our target image size
    # We need to convert it to a 3-channel image for the CNN
    img = np.stack([S_db_norm]*3, axis=-1)
    resized_img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    
    return resized_img

def prepare_augmented_data(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio'):
    """
    Loads audio, creates a balanced and augmented dataset of spectrograms.
    """
    if not os.path.exists(labels_file) or not os.path.isdir(audio_folder):
        print("Error: Ensure labels file and audio folder exist and paths are correct.")
        return None, None

    print("Loading mood labels...")
    labels_df = pd.read_csv(labels_file)

    print("Loading original audio data into memory...")
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

    print("\nCreating a balanced and augmented dataset of spectrograms...")
    all_spectrograms = []
    all_labels = []
    
    mood_counts = labels_df['mood'].value_counts()
    target_count = mood_counts.max()
    print(f"Balancing all classes to {target_count} samples each via augmentation.")

    for mood, count in mood_counts.items():
        print(f"Processing and augmenting mood: {mood}")
        mood_song_ids = [sid for sid, data in audio_data.items() if data['mood'] == mood]
        
        # Create a balanced list of song IDs, with duplicates for minority classes
        balanced_song_ids = mood_song_ids * (target_count // count)
        balanced_song_ids += random.sample(mood_song_ids, target_count % count)
        
        for song_id in balanced_song_ids:
            original_audio = audio_data[song_id]
            # Augment the audio data on the fly
            augmented_y = augment_audio(original_audio['y'], original_audio['sr'])
            # Create spectrogram from the (potentially) augmented data
            spectrogram = create_spectrogram_from_data(augmented_y, original_audio['sr'])
            
            all_spectrograms.append(spectrogram)
            all_labels.append(mood)

    return np.array(all_spectrograms), np.array(all_labels)

def train_augmented_cnn_model(X, y_text):
    """
    Trains the final CNN model on the augmented spectrogram data.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nBuilding the CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    print("\nTraining the CNN model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Augmented CNN Mood Model Accuracy: {accuracy * 100:.2f}%")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model.save('mood_cnn_augmented_model.keras')
    joblib.dump(label_encoder, 'mood_cnn_label_encoder.joblib')
    
    print("\nDone! Your final, most robust mood model is ready.")

if __name__ == "__main__":
    path_to_audio_files = '/home/j/AI-Genre-Tagger/'
    
    X_specs, y_labels = prepare_augmented_data(audio_folder=path_to_audio_files)
    
    if X_specs is not None:
        train_augmented_cnn_model(X_specs, y_labels)