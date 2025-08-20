# AI Music Genre & Mood Tagger

An AI-powered music organizer, built in Python, that analyzes audio files to predict their genre and emotional mood using a portfolio of custom-trained machine learning models.

This project serves as an end-to-end example of a machine learning workflow, demonstrating a journey from simple models to advanced deep learning architectures. It covers custom data preparation, model training with techniques like data augmentation and transfer learning, and a practical application for organizing a personal music library.

---

## 🎵 Features

* **Dual AI Models:** Classifies music based on two distinct, custom-trained models:
    * **Genre Classification:** A `scikit-learn` RandomForest model trained on a personal music collection to predict custom genres (e.g., Pop, Hip-Hop, Dubstep).
    * **Mood Analysis:** An advanced `TensorFlow` Neural Network, trained with **data augmentation** and **transfer learning** from Google's YAMNet, to predict emotional quadrants based on the Valence-Arousal model (e.g., Happy/Excited, Sad/Depressed).
* **Iterative Model Development:** The repository includes multiple training scripts, showcasing a progression from simple models to more complex and accurate architectures (RandomForest -> Dense NN -> CNN -> Transfer Learning).
* **Audio Feature Extraction:** Analyzes local `.mp3` or `.wav` files to extract key audio "fingerprints" using the `librosa` library.
* **Automated Library Classification:** Scans a folder of music and outputs a `.csv` file summarizing the entire library with predicted genres and moods, perfect for sorting and analysis.

---

## 🛠️ How It Works

The project is divided into two main workflows: training the models and using them to organize a library.

### 1. Model Training

The project includes a suite of training scripts that demonstrate different machine learning techniques:

* **Genre Model (`train_custom_model.py`):**
    * Learns from a directory of audio files that you have organized into subfolders by genre.
    * Extracts audio features and trains a **RandomForest classifier**.
    * Saves the trained model as `genre_classifier_model.joblib`.

* **Mood Models (A Progressive Journey):**
    * **Data Preparation (`explore_mood_data.py`):** Processes raw Valence-Arousal data into four distinct mood quadrants.
    * **Initial NN (`train_mood_nn.py`):** Trains a simple Dense Neural Network on an oversampled dataset.
    * **CNN Experiment (`train_mood_cnn_augmented.py`):** Trains a Convolutional Neural Network on augmented spectrogram "images" of the audio.
    * **Final Model (`train_mood_transfer_augmented.py`):** The most advanced model. It combines **data augmentation** with **transfer learning** from Google's expert YAMNet model to achieve the most robust and balanced performance. It saves the final model as `mood_transfer_augmented_model.keras`.

### 2. Music Organization (`music_organizer.py`)

* This script loads your pre-trained custom models for both genre and mood.
* It scans a user-specified folder of your personal music.
* For each track, it performs the necessary audio analysis to predict both the genre and the mood using the best-trained models.
* It compiles all the information into a final `my_music_library_classified.csv` report.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* A personal collection of audio files (`.mp3`, `.wav`) to train the models on.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/theouterlimitz/AI-Genre-Tagger.git](https://github.com/theouterlimitz/AI-Genre-Tagger.git)
    cd AI-Genre-Tagger
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage: The Workflow

1.  **Prepare Your Training Data:**
    * **For Genre:** Create a `training_data` folder with subfolders for each genre and fill them with representative audio files.
    * **For Mood:** Download a mood dataset (like the DEAM dataset) containing audio files and valence/arousal CSVs.

2.  **Train Your Custom Models:**
    * Run the desired training scripts (e.g., `train_custom_model.py` for genre, and `train_mood_transfer_augmented.py` for the best mood model).

3.  **Organize Your Music Library:**
    * Run the main organizer script. It will use the models you've trained and output a CSV file with both genre and mood predictions.
        ```bash
        python3 music_organizer.py
        ```

---

## ⚖️ Legal Disclaimer

The machine learning models are designed to be trained on the user's personal, non-distributed collection of legally obtained audio files for educational and research purposes. The audio files themselves are not included in this repository. The scripts are provided to allow users to create and use their own models on their own datasets.

---

## 🔮 Future Work

* **Implement Robust Tag-Writing:** Solve the library conflicts to add a feature that writes the predicted genres and moods directly back to the audio files' metadata.
* **Upgrade to Multi-Label Tagging:** Transition from a single-genre classifier to a multi-label model that can assign multiple tags (e.g., `['Pop', 'Electronic']`) to hybrid tracks.
* **Hyperparameter Tuning:** Implement `GridSearchCV` on the final transfer learning model to find the optimal settings for the classification head.
