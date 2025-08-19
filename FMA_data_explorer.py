# This script uses the pandas library to load and explore the metadata
# from the Free Music Archive (FMA) dataset. Its purpose is to help us
# understand the data we will use to train our machine learning model.

import pandas as pd
import os

def explore_fma_metadata(metadata_path):
    """
    Loads the FMA tracks.csv file and prints summary information.

    Args:
        metadata_path (str): The path to the fma_metadata directory.
    """
    # Construct the full path to the tracks.csv file
    tracks_csv_path = os.path.join(metadata_path, 'tracks.csv')

    if not os.path.exists(tracks_csv_path):
        print(f"Error: Could not find 'tracks.csv' at {tracks_csv_path}")
        print("Please make sure the 'fma_metadata' folder is in the same directory as this script.")
        return

    print(f"Loading track metadata from: {tracks_csv_path}\n")

    try:
        # Pandas can read CSVs with complex headers. We need to tell it
        # that the column names are in the first row (header=0) and the
        # track IDs are the first column (index_col=0).
        tracks = pd.read_csv(tracks_csv_path, index_col=0, header=[0, 1])
        
        # --- Filter for only the 'small' dataset ---
        # The CSV contains info for all dataset sizes, so we select just the 'small' subset.
        small_tracks = tracks[tracks['set', 'subset'] == 'small']

        print("--- Successfully loaded the data! ---\n")

        # --- 1. Show the first 5 rows of the data ---
        print("1. Here's a sample of the data (first 5 tracks):")
        # .head() is a great way to quickly peek at your data
        print(small_tracks[['track', 'artist', 'album', 'set']].head())
        print("\n" + "="*50 + "\n")

        # --- 2. Show the distribution of Genres ---
        print("2. How many songs per genre do we have?")
        # We access the 'genre_top' column and use .value_counts() to count them.
        genre_counts = small_tracks['track', 'genre_top'].value_counts()
        print(genre_counts)
        print("\nThis is great! We have a balanced set of 1000 songs for each of the 8 genres.")
        print("This will be a perfect starting point for our mood labels.")
        print("\n" + "="*50 + "\n")

        # --- 3. Show all available data columns ---
        print("3. What other information is available for each track?")
        # We can print the column names to see everything we could potentially use.
        # We'll just show the top-level column groups for simplicity.
        available_columns = small_tracks.columns.levels[0]
        print(available_columns)
        print("\nAs you can see, there's a ton of info, including track listens, interest, date, etc.")

    except Exception as e:
        print(f"An error occurred while processing the CSV file: {e}")


if __name__ == "__main__":
    # Define the path to the metadata directory.
    # This assumes the 'fma_metadata' folder is in the same directory as the script.
    path_to_metadata = 'fma_metadata'
    explore_fma_metadata(path_to_metadata)