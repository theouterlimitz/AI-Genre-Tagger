# This script trains a Convolutional Neural Network (CNN) to classify music moods
# by treating audio spectrograms as images.

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

def create_spectrogram(y, sr, file_path):
    """
    Creates and saves a Mel Spectrogram from audio data.
    """
    plt.interactive(False)
    fig = plt.figure(figsize=[1, 1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    
    fig.savefig(file_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def prepare_data(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio', spec_folder='spectrograms'):
    """
    Loads audio, creates spectrograms, and prepares data for the CNN.
    """
    if not os.path.exists(labels_file) or not os.path.isdir(audio_folder):
        print("Error: Ensure labels file and audio folder exist and paths are correct.")
        return None, None

    if not os.path.exists(spec_folder):
        os.makedirs(spec_folder)
        print(f"Created directory: {spec_folder}")

    labels_df = pd.read_csv(labels_file)
    
    all_spec_paths = []
    all_labels = []

    print("Creating and saving spectrograms...")
    for index, row in labels_df.iterrows():
        song_id = row['song_id']
        mood_label = row['mood']
        audio_path = os.path.join(audio_folder, f"{song_id}.mp3")
        spec_path = os.path.join(spec_folder, f"{song_id}.png")
        
        if os.path.exists(audio_path) and not os.path.exists(spec_path):
            try:
                y, sr = librosa.load(audio_path, mono=True, duration=30)
                create_spectrogram(y, sr, spec_path)
            except Exception as e:
                print(f"Could not process {audio_path}: {e}")
        
        if os.path.exists(spec_path):
            all_spec_paths.append(spec_path)
            all_labels.append(mood_label)

    return np.array(all_spec_paths), np.array(all_labels)

def train_mood_cnn_model(X_paths, y_text):
    """
    Trains a CNN model on the spectrogram data.
    """
    # --- 1. Encode Labels and Split Data ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(X_paths, y, test_size=0.2, random_state=42, stratify=y)

    # --- 2. Load Images ---
    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        return img / 255.0

    X_train = np.array([load_image(path) for path in X_train_paths])
    X_test = np.array([load_image(path) for path in X_test_paths])

    # --- 3. Build the CNN Model ---
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

    # --- 4. Train the Model ---
    print("\nTraining the CNN model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)

    # --- 5. Evaluate and Save ---
    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN Mood Model Accuracy: {accuracy * 100:.2f}%")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model.save('mood_cnn_model.keras')
    joblib.dump(label_encoder, 'mood_cnn_label_encoder.joblib')
    
    print("\nDone! Your CNN mood model is ready.")

if __name__ == "__main__":
    path_to_audio_files = '/home/j/AI-Genre-Tagger/'
    
    # Prepare data (this will create spectrograms if they don't exist)
    X_paths, y_text = prepare_data(audio_folder=path_to_audio_files)
    
    if X_paths is not None:
        train_mood_cnn_model(X_paths, y_text)