# This script uses GridSearchCV to find the best hyperparameters
# for our neural network mood classifier.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scikeras.wrappers import KerasClassifier
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
        print(f"Error processing {file_path}: {e}")
        return None

def create_model(optimizer='adam', dropout_rate=0.3, neurons=128):
    """
    Creates and compiles a Keras model with specified hyperparameters.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(neurons, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons // 2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def tune_mood_model(labels_file='processed_mood_labels.csv', audio_folder='path/to/your/audio'):
    """
    Loads data, extracts features, and performs hyperparameter tuning.
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
        file_path = os.path.join(audio_folder, f"{song_id}.mp3")
        if os.path.exists(file_path):
            features = extract_features_from_file(file_path)
            if features is not None:
                all_features.append(features)
                all_labels.append(row['mood'])

    X = np.array(all_features)
    y_text = np.array(all_labels)

    # --- 2. Encode Text Labels to Numbers ---
    global label_encoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    
    # --- 3. Split and Balance Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Balancing the training data with oversampling...")
    oversampler = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # --- 4. Scale Features ---
    global X_train_scaled
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # --- 5. Hyperparameter Tuning with GridSearchCV ---
    print("\nStarting hyperparameter tuning...")
    model = KerasClassifier(build_fn=create_model, verbose=0)
    
    param_grid = {
        'optimizer': ['adam', 'rmsprop'],
        'dropout_rate': [0.2, 0.3, 0.4],
        'neurons': [64, 128, 256]
    }
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train_scaled, y_train_resampled)

    # --- 6. Summarize Results ---
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    
    # --- 7. Save the Best Model ---
    best_model = grid_result.best_estimator_.model
    best_model.save('mood_nn_model_best.keras')
    joblib.dump(scaler, 'mood_nn_scaler_best.joblib')
    joblib.dump(label_encoder, 'mood_label_encoder_best.joblib')
    
    print("\nDone! Your best Neural Network mood model is ready.")

if __name__ == "__main__":
    # !!! IMPORTANT !!!
    # Update this path to your mood dataset's audio folder.
    path_to_audio_files = '/home/j/AI-Genre-Tagger/mood_audio_files' # Example path
    tune_mood_model(audio_folder=path_to_audio_files)
