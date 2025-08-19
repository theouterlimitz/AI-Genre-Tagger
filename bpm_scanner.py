# This script scans a directory of audio files, reads their existing metadata,
# analyzes them for missing information (like BPM), and saves everything
# into a structured CSV file for easy management.

import os
import csv
import librosa
import numpy as np
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TXXX, error

def get_audio_features(file_path):
    """
    Analyzes an audio file to determine its BPM and other features.

    Args:
        file_path (str): The full path to the audio file.

    Returns:
        dict: A dictionary containing the estimated BPM, or None if an error occurs.
    """
    try:
        y, sr = librosa.load(file_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        features = {
            'bpm': round(np.mean(tempo), 2)
            # --- PLACEHOLDER FOR ADVANCED ANALYSIS ---
            # This is where you would add calls to a machine learning model
            # to determine genre or mood from the audio itself.
            # 'genre_prediction': analyze_genre(y, sr),
            # 'mood_prediction': analyze_mood(y, sr),
        }
        return features
    except Exception as e:
        print(f"Could not analyze {os.path.basename(file_path)}. Error: {e}")
        return None

def update_bpm_tag(file_path, bpm):
    """
    Writes the BPM value to a custom 'TXXX:BPM' tag in the audio file.

    Args:
        file_path (str): The path to the audio file.
        bpm (float): The BPM value to write.
    """
    try:
        audio = ID3(file_path)
        # TXXX is a custom text frame. We create one for BPM.
        frame = TXXX(encoding=3, desc='BPM', text=str(bpm))
        audio.add(frame)
        audio.save()
        print(f"    -> BPM tag ({bpm}) written to {os.path.basename(file_path)}")
    except Exception as e:
        print(f"    -> Could not write BPM tag for {os.path.basename(file_path)}. Error: {e}")


def process_music_folder(folder_path):
    """
    Scans a folder for audio files, processes them, and saves data to a CSV.

    Args:
        folder_path (str): The path to the folder to be scanned.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return

    music_library = []
    print(f"\nScanning '{folder_path}' for audio files...\n")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mp3', '.wav')):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            song_data = {
                'filename': filename,
                'artist': 'Unknown',
                'title': 'Unknown',
                'album': 'Unknown',
                'genre': 'Unknown',
                'bpm': None
            }

            try:
                # --- Step 1: Read Existing Metadata ---
                audio = EasyID3(file_path)
                song_data['artist'] = audio.get('artist', ['Unknown'])[0]
                song_data['title'] = audio.get('title', [os.path.splitext(filename)[0]])[0]
                song_data['album'] = audio.get('album', ['Unknown'])[0]
                song_data['genre'] = audio.get('genre', ['Unknown'])[0]
                
                # Check for an existing BPM tag separately
                id3_tags = ID3(file_path)
                if 'TXXX:BPM' in id3_tags:
                    song_data['bpm'] = float(id3_tags['TXXX:BPM'].text[0])
                
            except error:
                print(f"    -> No ID3 tags found for {filename}. Will analyze audio.")
            except Exception as e:
                print(f"    -> Error reading tags for {filename}: {e}")

            # --- Step 2: Analyze if BPM is missing ---
            if song_data['bpm'] is None:
                print("    -> BPM not found in tags. Analyzing audio...")
                features = get_audio_features(file_path)
                if features:
                    song_data['bpm'] = features['bpm']
                    # --- Step 3: Write new BPM back to the file ---
                    update_bpm_tag(file_path, song_data['bpm'])
            else:
                print(f"    -> Found existing BPM: {song_data['bpm']}")

            music_library.append(song_data)

    # --- Step 4: Save all data to a CSV file ---
    if not music_library:
        print("No audio files found to process.")
        return
        
    output_csv_path = os.path.join(folder_path, 'music_library.csv')
    print(f"\n...Scan complete. Saving library data to {output_csv_path}")
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'title', 'artist', 'album', 'genre', 'bpm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(music_library)
        print("Successfully created music_library.csv!")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")


if __name__ == "__main__":
    path = input("Enter the path to your music folder: ")
    process_music_folder(path)
