# This script combines our two best techniques: transfer learning with YAMNet
# and data augmentation to create our final, most robust mood classifier.

import os
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# --- Configuration ---
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
TARGET_SAMPLE_RATE = 16000 # YAMNet expects 16kHz audio

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

def get_yamnet_embedding(y, model):
    """
    Gets the YAMNet embedding for a given audio waveform.
    """
    scores, embeddings, spectrogram = model(y)
    return np.mean(embeddings, axis=0)

def prepare_augmented_transfer_data(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio'):
    """
    Loads audio, creates a balanced and augmented dataset, and extracts YAMNet embeddings.
    """
    if not os.path.exists(labels_file) or not os.path.isdir(audio_folder):
        print("Error: Ensure labels file and audio folder exist and paths are correct.")
        return None, None

    print("Loading mood labels...")
    labels_df = pd.read_csv(labels_file)
    
    # Load the YAMNet model once
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)

    print("Loading original audio data into memory...")
    audio_data = {}
    for index, row in labels_df.iterrows():
        song_id = row['song_id']
        file_path = os.path.join(audio_folder, f"{song_id}.mp3")
        if os.path.exists(file_path):
            try:
                # Load and resample audio for YAMNet
                wav_data = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)[0]
                audio_data[song_id] = {'y': wav_data, 'sr': TARGET_SAMPLE_RATE, 'mood': row['mood']}
            except Exception as e:
                print(f"Could not load {file_path}: {e}")

    print("\nCreating balanced dataset and extracting YAMNet embeddings...")
    all_embeddings = []
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
            # Get the YAMNet embedding from the (potentially) augmented data
            embedding = get_yamnet_embedding(augmented_y, yamnet_model)
            
            all_embeddings.append(embedding)
            all_labels.append(mood)

    return np.array(all_embeddings), np.array(all_labels)

def train_final_model(X, y_text):
    """
    Trains the final classification head on the augmented YAMNet embeddings.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nBuilding the final transfer learning model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],), dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    print("\nTraining the final classification head...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    print("\nEvaluating final model performance...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Mood Model Accuracy: {accuracy * 100:.2f}%")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model.save('mood_transfer_augmented_model.keras')
    joblib.dump(label_encoder, 'mood_transfer_label_encoder.joblib')
    
    print("\nDone! Your final, most robust mood model is ready.")

if __name__ == "__main__":
    path_to_audio_files = '/home/j/AI-Genre-Tagger/'
    
    X_embeddings, y_labels = prepare_augmented_transfer_data(audio_folder=path_to_audio_files)
    
    if X_embeddings is not None and len(X_embeddings) > 0:
        train_final_model(X_embeddings, y_labels)
