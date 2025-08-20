# This script processes the arousal and valence data, calculates the average
# scores for each song, assigns them to a mood quadrant, and saves the
# result to a new CSV file for our model training.

import pandas as pd
import os

def assign_mood_quadrant(valence, arousal):
    """
    Assigns a mood label based on valence and arousal scores.
    A score >= 0 is considered "high", < 0 is "low".
    """
    if valence >= 0 and arousal >= 0:
        return 'Happy/Excited'
    elif valence >= 0 and arousal < 0:
        return 'Calm/Content'
    elif valence < 0 and arousal >= 0:
        return 'Angry/Anxious'
    else: # valence < 0 and arousal < 0
        return 'Sad/Depressed'

def process_mood_data(arousal_path='arousal.csv', valence_path='valence.csv'):
    """
    Loads, processes, and merges the mood data into a single labeled file.
    """
    if not all(os.path.exists(p) for p in [arousal_path, valence_path]):
        print(f"Error: Make sure '{arousal_path}' and '{valence_path}' are in the same directory as this script.")
        return

    print("Loading arousal and valence data...")
    try:
        arousal_df = pd.read_csv(arousal_path)
        valence_df = pd.read_csv(valence_path)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # --- 1. Calculate Mean Scores ---
    print("Calculating average scores for each song...")
    # Set song_id as the index for easier calculations
    arousal_df.set_index('song_id', inplace=True)
    valence_df.set_index('song_id', inplace=True)

    # Calculate the mean across all sample columns for each song
    mean_arousal = arousal_df.mean(axis=1)
    mean_valence = valence_df.mean(axis=1)

    # --- 2. Merge Data into a Single DataFrame ---
    mood_df = pd.DataFrame({
        'mean_valence': mean_valence,
        'mean_arousal': mean_arousal
    })

    # --- 3. Assign Mood Quadrant Labels ---
    print("Assigning mood quadrant labels...")
    mood_df['mood'] = mood_df.apply(lambda row: assign_mood_quadrant(row['mean_valence'], row['mean_arousal']), axis=1)

    print("\n--- Data Processing Complete! ---\n")

    # --- 4. Display Summary ---
    print("1. Here's a sample of the processed mood data:")
    print(mood_df.head())
    print("\n" + "="*50 + "\n")

    print("2. How many songs per mood quadrant do we have?")
    print(mood_df['mood'].value_counts())
    print("\n" + "="*50 + "\n")

    # --- 5. Save the Processed Data ---
    output_filename = 'processed_mood_labels.csv'
    print(f"Saving the final labeled data to '{output_filename}'...")
    try:
        mood_df.to_csv(output_filename)
        print("Successfully created processed_mood_labels.csv!")
    except Exception as e:
        print(f"Could not save CSV file. Error: {e}")


if __name__ == "__main__":
    process_mood_data()
