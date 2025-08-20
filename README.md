# AI Music Genre & Mood Tagger

An AI-powered music organizer, built in Python, that analyzes audio files to predict their genre and emotional mood using custom-trained machine learning models.

This project serves as an end-to-end example of a machine learning workflow, from custom data preparation and model training to a practical, real-world application for organizing and understanding a personal music library.

---

## 🎵 Features

* **Dual AI Models:** Classifies music based on two separate, custom-trained models:
    * **Genre Classification:** Predicts the musical genre (e.g., Pop, Hip-Hop, Rock).
    * **Mood Analysis:** Predicts the emotional quadrant based on the Valence-Arousal model (e.g., Happy/Excited, Sad/Depressed).
* **Custom Model Training:** Learn from your own music! The tool includes scripts to train both genre and mood models on user-provided datasets.
* **Audio Feature Extraction:** Analyzes local `.mp3` or `.wav` files to extract key audio "fingerprints" using the `librosa` library.
* **Automated Library Classification:** Scans a folder of music and outputs a `.csv` file summarizing the entire library with predicted genres and moods, perfect for sorting and analysis.

---

## 🛠️ How It Works

The project is divided into two main workflows: training the models and using them to organize a library.

### 1. Model Training

The project uses two separate training scripts:

* **Genre Model (`train_custom_model.py`):**
    * Learns from a directory of audio files that you have organized into subfolders by genre (e.g., `/training_data/Pop`, `/training_data/Dubstep`).
    * Extracts audio features from each file and trains a model to associate those features with your genre labels.
    * Saves the trained model as `genre_classifier_model.joblib`.

* **Mood Model (`mood_classifier_trainer.py`):**
    * Learns from a dataset of audio files and corresponding Valence-Arousal data.
    * First, it processes the raw data into four mood quadrants (Happy, Sad, Angry, Calm).
    * It then extracts audio features from each file and trains a model to associate those features with the mood labels.
    * Saves the trained model as `mood_classifier_model.joblib`.

### 2. Music Organization (`music_organizer.py`)

* This script loads your pre-trained custom models for both genre and mood.
* It scans a user-specified folder of your personal music.
* For each track, it performs audio analysis to predict both the genre and the mood.
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
    * **For Genre:** Create a `training_data` folder with subfolders for each genre (e.g., `Pop`, `Rock`) and fill them with representative audio files.
    * **For Mood:** Download a mood dataset (like the DEAM dataset) containing audio files and valence/arousal CSVs.

2.  **Train Your Custom Models:**
    * Run the genre training script:
        ```bash
        python3 train_custom_model.py
        ```
    * Run the mood training script (after updating the audio path inside the file):
        ```bash
        python3 mood_classifier_trainer.py
        ```

3.  **Organize Your Music Library:**
    * Run the main organizer script. It will use the models you just trained and output a CSV file with both genre and mood predictions.
        ```bash
        python3 music_organizer.py
        ```

---

## ⚖️ Legal Disclaimer

The machine learning models are designed to be trained on the user's personal, non-distributed collection of legally obtained audio files for educational and research purposes. The audio files themselves are not included in this repository. The scripts are provided to allow users to create and use their own models on their own datasets.

---

## 🔮 Future Work

* **Implement Robust Tag-Writing:** Add a feature to write the predicted genres and moods directly back to the audio files' metadata.
* **Upgrade to Multi-Label Tagging:** Transition from a single-genre classifier to a multi-label model that can assign multiple tags (e.g., `['Pop', 'Electronic']`) to hybrid tracks.
* **Improve Model Accuracy:** Experiment with different model architectures and techniques for handling imbalanced data to improve prediction accuracy.
