# This script trains a machine learning model to classify music genres
# based on the pre-computed audio features from the FMA dataset.
# It then saves the trained model to a file for later use.

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(metadata_path):
    """
    Loads FMA features and tracks, trains a genre classifier, and saves it.

    Args:
        metadata_path (str): The path to the fma_metadata directory.
    """
    # --- 1. Load the Datasets ---
    tracks_csv_path = os.path.join(metadata_path, 'tracks.csv')
    features_csv_path = os.path.join(metadata_path, 'features.csv')

    if not all(os.path.exists(p) for p in [tracks_csv_path, features_csv_path]):
        print("Error: Ensure 'tracks.csv' and 'features.csv' are in the metadata folder.")
        return

    print("Loading tracks and features data...")
    try:
        tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])
        features = pd.read_csv(features_csv_path, index_col=0, header=[0, 1, 2])
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # --- 2. Prepare the Data ---
    print("Preparing data for training...")
    # Filter for the 'small' subset and get the genre labels (our 'y')
    small_tracks = tracks[tracks['set', 'subset'] == 'small']
    y = small_tracks['track', 'genre_top']
    
    # Get the corresponding audio features (our 'X')
    # .loc is used to select rows from 'features' based on the index of 'small_tracks'
    X = features.loc[small_tracks.index]

    print(f"Data shape: {X.shape[0]} tracks and {X.shape[1]} features.")
    print(f"Target labels (genres): {y.unique().tolist()}")

    # --- 3. Split Data into Training and Testing Sets ---
    # We'll train the model on 80% of the data and test its performance on the other 20%.
    # random_state ensures we get the same split every time we run the script.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Scale the Features ---
    # ML models work best when all features are on a similar scale.
    # StandardScaler standardizes features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- 5. Train the Machine Learning Model ---
    print("\nTraining the Random Forest classifier... (This may take a minute or two)")
    # A RandomForest is a great, powerful, all-purpose classifier.
    # n_estimators is the number of "trees" in the forest.
    # n_jobs=-1 uses all available CPU cores to speed up training.
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_train_scaled, y_train)
    print("Training complete!")

    # --- 6. Evaluate the Model ---
    print("\nEvaluating model performance on the test set...")
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nThis means the model correctly guessed the genre on the test data about this often.")
    print("An accuracy above 50-60% on the first try is very good for this kind of problem!")

    # --- 7. Save the Trained Model and the Scaler ---
    model_filename = 'genre_classifier_model.joblib'
    scaler_filename = 'feature_scaler.joblib'
    
    print(f"\nSaving trained model to '{model_filename}'")
    joblib.dump(classifier, model_filename)
    
    print(f"Saving feature scaler to '{scaler_filename}'")
    joblib.dump(scaler, scaler_filename)
    
    print("\nDone! We now have a trained 'brain' ready to be used in our organizer script.")


if __name__ == "__main__":
    path_to_metadata = 'fma_metadata'
    train_model(path_to_metadata)