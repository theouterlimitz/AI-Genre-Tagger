# This script uses transfer learning with the pre-trained YAMNet model
# to create a highly efficient and powerful mood classifier.

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# --- YAMNet Configuration ---
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
TARGET_SAMPLE_RATE = 16000 # YAMNet expects 16kHz audio

def load_wav_16k_mono(file_path):
    """
    Loads a WAV file, converts it to mono, and resamples to 16kHz.
    """
    wav, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    return wav

def prepare_transfer_data(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio'):
    """
    Loads audio, gets YAMNet embeddings, and prepares data for training.
    """
    if not os.path.exists(labels_file) or not os.path.isdir(audio_folder):
        print("Error: Ensure labels file and audio folder exist and paths are correct.")
        return None, None

    print("Loading mood labels...")
    labels_df = pd.read_csv(labels_file)
    
    # Load the YAMNet model from TensorFlow Hub
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)

    all_embeddings = []
    all_labels = []

    print("Extracting YAMNet embeddings... (This may take some time)")
    for index, row in labels_df.iterrows():
        song_id = row['song_id']
        mood_label = row['mood']
        file_path = os.path.join(audio_folder, f"{song_id}.mp3")

        if os.path.exists(file_path):
            try:
                # Load and process the audio for YAMNet
                wav_data = load_wav_16k_mono(file_path)
                # Get the embeddings and scores from YAMNet
                scores, embeddings, spectrogram = yamnet_model(wav_data)
                # We use the mean of the embeddings over the whole clip as our feature
                clip_embedding = np.mean(embeddings, axis=0)
                
                all_embeddings.append(clip_embedding)
                all_labels.append(mood_label)
            except Exception as e:
                print(f"Could not process {file_path}: {e}")

    return np.array(all_embeddings), np.array(all_labels)

def train_transfer_model(X, y_text):
    """
    Trains a new classification head on top of the frozen YAMNet embeddings.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nBuilding the transfer learning model...")
    model = tf.keras.Sequential([
        # The input shape is the shape of one YAMNet embedding vector
        tf.keras.layers.Input(shape=(X_train.shape[1],), dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    print("\nTraining the new classification head...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Transfer Learning Mood Model Accuracy: {accuracy * 100:.2f}%")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model.save('mood_transfer_model.keras')
    joblib.dump(label_encoder, 'mood_transfer_label_encoder.joblib')
    
    print("\nDone! Your transfer learning mood model is ready.")

if __name__ == "__main__":
    path_to_audio_files = '/home/j/AI-Genre-Tagger/'
    
    # This step extracts the embeddings using YAMNet
    X_embeddings, y_labels = prepare_transfer_data(audio_folder=path_to_audio_files)
    
    if X_embeddings is not None and len(X_embeddings) > 0:
        # This step trains our new model on those embeddings
        train_transfer_model(X_embeddings, y_labels)