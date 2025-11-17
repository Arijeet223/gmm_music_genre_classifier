import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import os

def train_gmm_models(X, y, n_components=4):
    """
    Train a separate GMM for each genre.
    Uses regularization to avoid covariance collapse.
    """
    models = {}
    genres = np.unique(y)

    for genre in genres:
        print(f"🎧 Training GMM for genre: {genre}")

        # Get samples for this genre
        X_genre = X[y == genre].astype(np.float64)   # Force float64 for stability

        # Create GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            reg_covar=1e-3,       # prevents singular covariance matrices
            max_iter=300,
            n_init=3,
            random_state=42
        )

        gmm.fit(X_genre)
        models[genre] = gmm

    print("\n✅ GMM training complete.")
    return models


def save_models(models, path="models/gmm_genres.pkl"):
    os.makedirs("models", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)
    print(f"💾 Models saved to {path}")
