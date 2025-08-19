# AI Music Genre Tagger

An AI-powered music organizer, built in Python, that analyzes audio files to predict and tag their genre using a custom-trained machine learning model.

This project serves as an end-to-end example of a machine learning workflow, from custom data preparation and model training to a practical, real-world application for organizing a personal music library.

---

## 🎵 Features

* **Custom Model Training:** Learn from your own music! The tool trains a model based on audio files you provide, organized into genre-specific folders.
* **Audio Feature Extraction:** Analyzes local `.mp3` or `.wav` files to extract key audio "fingerprints" using the `librosa` library.
* **ML-Powered Genre Prediction:** Uses a `scikit-learn` RandomForest model to predict the genre of unknown tracks.
* **Automated Metadata Tagging:** Reads existing ID3 tags and writes new, predicted genre tags directly back to your audio files using `mutagen`.
* **Library Reporting:** Scans a folder of music and outputs a `.csv` file summarizing the entire library with the final genre classifications.

---

## 🛠️ How It Works

This project is divided into two main parts, designed to create a personalized genre classifier:

1.  **Model Training (`train_custom_model.py`):**
    * The script processes a directory of audio files that you have organized into subfolders by genre (e.g., `/training_data/Pop`, `/training_data/Dubstep`).
    * It extracts a comprehensive set of audio features from each file to understand its acoustic properties.
    * It then trains a RandomForest classifier to learn the specific patterns that connect your audio features to your genre labels.
    * Finally, it saves the trained model (`genre_classifier_model.joblib`) and a feature scaler (`feature_scaler.joblib`) for later use.

2.  **Music Organizer (`music_organizer.py`):**
    * This script loads your pre-trained custom model.
    * It scans a different, user-specified folder of your personal music (e.g., your main library or a folder of new tracks).
    * For each track, it first checks for an existing genre tag.
    * If the genre is missing or unknown, it extracts the audio features, feeds them to your model for a prediction, and writes the predicted genre back into the file's ID3 tags.
    * It compiles all the information into a final `my_music_library_classified.csv` report.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* A personal collection of audio files (`.mp3`, `.wav`) to train the model on.

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

### Usage: The 3-Step Workflow

1.  **Prepare Your Training Data:**
    * Create a folder named `training_data` inside the project directory.
    * Inside it, create subfolders for each genre you want to classify (e.g., `Pop`, `Rock`, `Dubstep`, `Hip-Hop`).
    * Place at least 15-20 representative audio files into each genre subfolder. **The quality and diversity of this data is the most important factor for model accuracy.**

2.  **Train Your Custom Model:**
    * Run the custom training script. This will create (or overwrite) the `genre_classifier_model.joblib` and `feature_scaler.joblib` files.
    ```bash
    python3 train_custom_model.py
    ```

3.  **Organize Your Music Library:**
    * Run the main organizer script and provide the path to your music collection when prompted. The script will use the model you just trained.
    ```bash
    python3 music_organizer.py
    ```

---

## ⚖️ Legal Disclaimer

The machine learning model is designed to be trained on the user's personal, non-distributed collection of legally obtained audio files for educational and research purposes. The audio files themselves are not included in this repository. The scripts are provided to allow users to create and use their own models on their own datasets.

---

## 🔮 Future Work

* **Upgrade to Multi-Label Tagging:** Transition from a single-genre classifier to a multi-label model that can assign multiple tags (e.g., `['Pop', 'Electronic']`) to hybrid tracks.
* **Mood & Emotion Analysis:** Expand the model to predict moods based on the Valence-Arousal model (e.g., Happy, Sad, Energetic, Calm).
