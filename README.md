# AI Music Genre Classifier

A Python-based tool that uses machine learning to analyze audio files, predict their genre, and organize a music library. This project serves as an end-to-end example of a machine learning workflow, from data preparation and model training to a practical, real-world application.

---

## 🎵 Features

* **Audio Feature Extraction:** Analyzes local `.mp3` or `.wav` files to extract key audio features using the `librosa` library.
* **Metadata Management:** Reads and writes ID3 tags (Artist, Title, Genre) using the `mutagen` library.
* **ML-Powered Genre Prediction:** Uses a `scikit-learn` RandomForest model trained on a custom dataset to predict the genre of unknown tracks.
* **Automated Library Organization:** Scans a folder of music and outputs a `.csv` file summarizing the entire library with predicted genres.
* **Custom Model Training:** Includes a separate script to train the classification model on a user-provided dataset of audio files.

---

## 🛠️ How It Works

This project is divided into two main parts:

1.  **Model Training (`train_genre_classifier.py`):**
    * The script takes a directory of audio files, organized into subfolders by genre (e.g., `/training_data/Pop`, `/training_data/Dubstep`).
    * It extracts a comprehensive set of audio features from each file.
    * It then trains a RandomForest classifier to learn the relationship between the audio features and their corresponding genre labels.
    * Finally, it saves the trained model (`.joblib` file) for future use.

2.  **Music Organizer (`music_organizer.py`):**
    * This script loads the pre-trained model.
    * It scans a user-specified folder of personal music.
    * For each track, it first checks for an existing genre tag.
    * If the genre is missing, it extracts the audio features, feeds them to the model for a prediction, and writes the predicted genre back into the file's ID3 tags.
    * It compiles all the information into a final `music_library.csv` report.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* A collection of audio files (`.mp3`, `.wav`) to train the model on.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/[YourRepoName].git
    cd [YourRepoName]
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
    *(Note: You will need to create a `requirements.txt` file. You can do this by running `pip freeze > requirements.txt` in your activated environment after installing all the packages we've used.)*

### Usage

1.  **Prepare Your Training Data:**
    * Create a folder named `training_data`.
    * Inside it, create subfolders for each genre you want to classify (e.g., `Pop`, `Rock`, `Hip-Hop`).
    * Place at least 15-20 representative audio files into each genre subfolder.

2.  **Train Your Custom Model:**
    * Run the training script. This will create the `genre_classifier_model.joblib` file.
    ```bash
    python3 train_genre_classifier.py
    ```

3.  **Organize Your Music Library:**
    * Run the main organizer script and provide the path to your music collection when prompted.
    ```bash
    python3 music_organizer.py
    ```

---

## ⚖️ Legal Disclaimer

The machine learning model included in this project was trained on a personal, non-distributed collection of legally obtained audio files for educational and research purposes under the doctrine of Fair Use. The audio files themselves are not included in this repository. The scripts are provided to allow users to train their own models on their own datasets.

---

## 🔮 Future Work

* **Upgrade to Multi-Label Tagging:** Transition from a single-genre classifier to a multi-label model that can assign multiple tags (e.g., `['Pop', 'Electronic']`) to hybrid tracks.
* **Mood & Emotion Analysis:** Expand the model to predict moods based on the Valence-Arousal model (e.g., Happy, Sad, Energetic, Calm).
