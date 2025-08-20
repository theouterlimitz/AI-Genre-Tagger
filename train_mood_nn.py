# This script trains a Neural Network to classify music moods.
# It uses TensorFlow (Keras) to build a more sophisticated model
# capable of learning complex patterns in the audio features.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import joblib
import librosa
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

def extract_features_from_file(file_path):
    """
    Extracts a set of audio features from a single audio file.
    """
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        # Extract a robust set of features
        features_to_extract = [
            librosa.feature.chroma_stft, librosa.feature.rms,
            librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth,
            librosa.feature.spectral_rolloff, librosa.feature.zero_crossing_rate
        ]
        
        all_features = []
        for func in features_to_extract:
            # Check if the function requires the sample rate argument
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
        print(f"Error processing {file_path}: {e}")
        return None

def train_mood_nn_model(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio'):
    """
    Loads data, extracts features, and trains a neural network for mood classification.
    """
    if not os.path.exists(labels_file) or not os.path.isdir(audio_folder):
        print("Error: Ensure labels file and audio folder exist and paths are correct.")
        return

    # --- 1. Load Data and Extract Features ---
    print("Loading mood labels...")
    labels_df = pd.read_csv(labels_file, index_col='song_id')

    all_features = []
    all_labels = []
    
    print("Starting feature extraction...")
    for song_id, row in labels_df.iterrows():
        # NOTE: You may need to change '.mp3' if your files are '.wav', etc.
        file_path = os.path.join(audio_folder, f"{song_id}.mp3")
        if os.path.exists(file_path):
            features = extract_features_from_file(file_path)
            if features is not None:
                all_features.append(features)
                all_labels.append(row['mood'])

    X = np.array(all_features)
    y_text = np.array(all_labels)

    # --- 2. Encode Text Labels to Numbers ---
    print("Encoding text labels to numbers...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    # --- 3. Split and Balance Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Balancing the training data with oversampling...")
    oversampler = RandomOverSampler(random_state=42)
    # --- THIS IS THE CORRECTED LINE ---
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # --- 4. Scale Features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. Build and Train the Neural Network ---
    print("\nBuilding the Neural Network model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax') # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training the model...")
    model.fit(X_train_scaled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.1, verbose=2)

    # --- 6. Evaluate the Model ---
    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Mood Model Accuracy: {accuracy * 100:.2f}%")
    
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # --- 7. Save the Model, Scaler, and Label Encoder ---
    model.save('mood_nn_model.keras')
    joblib.dump(scaler, 'mood_nn_scaler.joblib')
    joblib.dump(label_encoder, 'mood_label_encoder.joblib')
    
    print("\nDone! Your Neural Network mood model is ready.")

if __name__ == "__main__":
    # !!! IMPORTANT !!!
    # Update this path to your mood dataset's audio folder.
    path_to_audio_files = '/home/j/AI-Genre-Tagger/' # Example path
    train_mood_nn_model(audio_folder=path_to_audio_files)
