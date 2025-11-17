import librosa
import numpy as np
import os

def extract_mfcc(file_path, n_mfcc=20):
    """
    Extract MFCC features from an audio file.
    Returns the mean MFCC vector for the entire audio clip.
    """
    # Load audio (librosa handles resampling internally)
    y, sr = librosa.load(file_path, duration=30)

    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Return mean across time frames
    return np.mean(mfcc.T, axis=0)


def extract_dataset_features(dataset_path):
    """
    Extract features for all audio files in all genre folders.
    Automatically skips corrupted audio files.
    """
    features = []
    labels = []

    genres = os.listdir(dataset_path)

    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)

        # Ensure this is a directory
        if not os.path.isdir(genre_path):
            continue

        print(f"\n🎵 Processing genre: {genre}")

        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)

            # Ensure it's a file
            if not os.path.isfile(file_path):
                continue

            try:
                mfcc = extract_mfcc(file_path)
                features.append(mfcc)
                labels.append(genre)

            except Exception as e:
                # Skip corrupted / unreadable file
                print(f"⚠️ Skipping corrupted file: {file_path}")
                continue

    print("\n✅ Feature extraction complete.")
    print(f"Total usable samples: {len(features)}")

    return np.array(features), np.array(labels)
