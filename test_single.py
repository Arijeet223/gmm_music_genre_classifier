import pickle
from src.extract_features import extract_mfcc
from src.evaluate import predict

# Load trained models
with open("models/gmm_genres.pkl", "rb") as f:
    models = pickle.load(f)

# Path to your audio file
file_path = "test_song.wav"

# Extract MFCC
mfcc = extract_mfcc(file_path)

# Predict
genre = predict(models, mfcc)

print("Predicted Genre:", genre)
