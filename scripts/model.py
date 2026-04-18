"""
scripts/model.py
----------------
Trains and saves all three Group Rec models:

  1. PopularityRecommender  - naive baseline (most-rated films)
  2. SVDRecommender         - classical matrix factorization (Surprise)
  3. NCFRecommender         - neural collaborative filtering (PyTorch)

Each class implements a consistent interface:
  .fit(train_df)
  .predict(user_id, movie_ids) -> Dict[movie_id, score]
  .save(path)
  .load(path)   [classmethod]

Usage:
    python scripts/model.py
"""

import os
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from surprise import SVD, Dataset as SurpriseDataset, Reader

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# Rating scale constants - used for normalization throughout
RATING_MIN = 0.5
RATING_MAX = 5.0
RATING_RANGE = RATING_MAX - RATING_MIN


def normalize_ratings(ratings: torch.Tensor) -> torch.Tensor:
    """
    Normalize ratings from [0.5, 5.0] to [0.0, 1.0].

    Normalization stabilizes training by keeping target values in the
    same range as the sigmoid output, which prevents gradient saturation.

    Args:
        ratings: tensor of raw ratings in [0.5, 5.0]

    Returns:
        tensor of normalized ratings in [0.0, 1.0]
    """
    return (ratings - RATING_MIN) / RATING_RANGE


def denormalize_ratings(normalized: np.ndarray) -> np.ndarray:
    """
    Scale normalized predictions from [0.0, 1.0] back to [0.5, 5.0].

    Args:
        normalized: array of sigmoid outputs in [0.0, 1.0]

    Returns:
        array of predicted ratings in [0.5, 5.0]
    """
    return normalized * RATING_RANGE + RATING_MIN


# ---------------------------------------------------------------------------
# 1. Naive Baseline - Popularity Recommender
# ---------------------------------------------------------------------------

class PopularityRecommender:
    """
    Naive baseline recommender.

    Recommends the globally highest average-rated movies (minimum rating
    count threshold applied to avoid rating gaming by obscure films).
    No personalization - same ranked list for every user.

    This is the honest lower bound: how much better does personalization do?
    """

    MIN_RATING_COUNT = 500

    def __init__(self):
        self.popular_movies: pd.DataFrame = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Compute popularity scores from training ratings.

        Args:
            train_df: DataFrame with columns [userId, movieId, rating]
                userId       (int)   - MovieLens user identifier
                movieId      (int)   - MovieLens movie identifier
                rating       (float) - user rating in [0.5, 5.0]
        """
        stats = (
            train_df.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stats.columns = ["movieId", "avg_rating", "rating_count"]
        filtered = stats[stats["rating_count"] >= self.MIN_RATING_COUNT]
        self.popular_movies = filtered.sort_values("avg_rating", ascending=False)

    def predict(self, user_id: int, movie_ids: List[int]) -> Dict[int, float]:
        """
        Return popularity scores for each movie_id.
        Score is the global average rating (same for all users).

        Args:
            user_id:   ignored for this baseline - no personalization
            movie_ids: list of candidate movie IDs to score

        Returns:
            Dict mapping movie_id -> global average rating score
        """
        score_map = dict(zip(
            self.popular_movies["movieId"],
            self.popular_movies["avg_rating"]
        ))
        return {mid: score_map.get(mid, 0.0) for mid in movie_ids}

    def get_top_movies(self, top_k: int = 100) -> List[int]:
        """Return the top_k most popular movie IDs by average rating."""
        return self.popular_movies["movieId"].head(top_k).tolist()

    def save(self, path: str) -> None:
        """Save model to disk using joblib."""
        joblib.dump(self, path)
        print(f"  Saved PopularityRecommender -> {path}")

    @classmethod
    def load(cls, path: str) -> "PopularityRecommender":
        """Load model from disk."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# 2. Classical Model - SVD Matrix Factorization
# ---------------------------------------------------------------------------

class SVDRecommender:
    """
    Classical matrix factorization recommender using SVD.

    Built on the Surprise library's SVD implementation, which follows
    the approach from the Netflix Prize (Koren, 2009). Learns latent
    user and item factor vectors that minimize rating prediction error.

    Hyperparameters tuned via grid search:
        n_factors: number of latent dimensions (50, 100, 200)
        lr_all:    learning rate for all parameters
        reg_all:   L2 regularization strength for all parameters
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr_all: float = 0.005, reg_all: float = 0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model: Optional[SVD] = None
        self.trainset = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train SVD on the ratings dataframe.

        Args:
            train_df: DataFrame with columns [userId, movieId, rating]
                userId   (int)   - MovieLens user identifier
                movieId  (int)   - MovieLens movie identifier
                rating   (float) - user rating in [0.5, 5.0]
        """
        print(f"  Training SVD (n_factors={self.n_factors}, "
              f"n_epochs={self.n_epochs})...")
        reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
        data = SurpriseDataset.load_from_df(
            train_df[["userId", "movieId", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=False,
        )
        self.model.fit(self.trainset)
        print("  SVD training complete.")

    def predict(self, user_id: int, movie_ids: List[int]) -> Dict[int, float]:
        """
        Predict ratings for a user across a list of movie IDs.

        Args:
            user_id:   MovieLens user ID
            movie_ids: candidate movie IDs to score

        Returns:
            Dict mapping movie_id -> predicted rating in [0.5, 5.0]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .fit() first.")
        return {
            mid: self.model.predict(user_id, mid).est
            for mid in movie_ids
        }

    def evaluate_rmse(self, test_df: pd.DataFrame) -> float:
        """
        Compute RMSE on a held-out test set.

        Args:
            test_df: DataFrame with columns [userId, movieId, rating]

        Returns:
            Root mean squared error as a float
        """
        errors = []
        for _, row in test_df.iterrows():
            pred = self.model.predict(row["userId"], row["movieId"]).est
            errors.append((pred - row["rating"]) ** 2)
        return float(np.sqrt(np.mean(errors)))

    def save(self, path: str) -> None:
        """Save model to disk using joblib."""
        joblib.dump(self, path)
        print(f"  Saved SVDRecommender -> {path}")

    @classmethod
    def load(cls, path: str) -> "SVDRecommender":
        """Load model from disk."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# 3. Deep Learning Model - Neural Collaborative Filtering (NCF)
# ---------------------------------------------------------------------------

class RatingsDataset(Dataset):
    """
    PyTorch Dataset for user-item rating pairs.

    Stores integer indices for users and movies (not raw IDs) and
    normalized ratings in [0.0, 1.0] for stable sigmoid training.
    """

    def __init__(self, df: pd.DataFrame):
        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.movie_idx = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.user_idx[idx], self.movie_idx[idx], self.ratings[idx]


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering model (He et al., 2017).

    Architecture combines two paths:
      - GMF path: element-wise product of user and item embeddings
        captures linear interaction signals between users and items
      - MLP path: concatenated embeddings passed through fully connected
        layers to capture non-linear interaction patterns
      - Output: both paths fused via a linear layer, passed through
        sigmoid to produce a value in [0, 1], then scaled to [0.5, 5.0]

    Using sigmoid instead of clamp is critical - clamp produces zero
    gradients at the boundaries which prevents the model from learning.
    Sigmoid has a smooth, non-zero gradient everywhere.

    Reference:
      He, X. et al. (2017). Neural Collaborative Filtering. WWW 2017.
    """

    def __init__(self, n_users: int, n_movies: int,
                 emb_dim: int = 64, mlp_layers: List[int] = None,
                 dropout: float = 0.2):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        # GMF embeddings - one vector per user and item for dot-product path
        self.gmf_user_emb = nn.Embedding(n_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(n_movies, emb_dim)

        # MLP embeddings - separate vectors for the deep path
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_movies, emb_dim)

        # MLP layers - progressively compress the concatenated embedding
        mlp_input_dim = emb_dim * 2
        layers = []
        for out_dim in mlp_layers:
            layers += [nn.Linear(mlp_input_dim, out_dim), nn.ReLU(),
                       nn.Dropout(dropout)]
            mlp_input_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # Output layer fuses GMF element-wise product with MLP final output
        self.output_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform initialization for all embedding and linear layers.
        Keeps initial activations in a stable range to prevent vanishing
        or exploding gradients at the start of training.
        """
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, user_idx: torch.Tensor,
                movie_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both GMF and MLP paths.

        Args:
            user_idx:  (batch_size,) tensor of integer user indices
            movie_idx: (batch_size,) tensor of integer movie indices

        Returns:
            (batch_size,) tensor of predicted ratings in [0.5, 5.0]

        Note:
            sigmoid maps output to [0, 1], then we scale to [0.5, 5.0].
            This is preferred over clamp because sigmoid has a non-zero
            gradient everywhere, allowing the model to keep learning even
            when predictions are near the boundaries.
        """
        # GMF path: element-wise product captures linear co-occurrence signal
        gmf_user = self.gmf_user_emb(user_idx)
        gmf_item = self.gmf_item_emb(movie_idx)
        gmf_out = gmf_user * gmf_item

        # MLP path: concatenation + deep layers captures non-linear patterns
        mlp_user = self.mlp_user_emb(user_idx)
        mlp_item = self.mlp_item_emb(movie_idx)
        mlp_in = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Fusion: concatenate both path outputs and project to scalar
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.output_layer(fused)

        # Sigmoid maps to [0, 1], then scale to [0.5, 5.0]
        return torch.sigmoid(output.squeeze(-1)) * RATING_RANGE + RATING_MIN


class NCFRecommender:
    """
    Wrapper around NCFModel for training, prediction, and persistence.

    Handles data preparation, training loop, inference, and model
    serialization. This is the deployed model in the Group Rec application.
    """

    def __init__(self, emb_dim: int = 64, mlp_layers: List[int] = None,
                 dropout: float = 0.2, lr: float = 1e-3,
                 batch_size: int = 1024, n_epochs: int = 10):
        """
        Args:
            emb_dim:    size of each user and item embedding vector
            mlp_layers: list of hidden layer sizes for the MLP path
            dropout:    dropout probability applied after each MLP layer
            lr:         Adam optimizer learning rate
            batch_size: number of (user, item, rating) triples per batch
            n_epochs:   number of full passes over the training data
        """
        self.emb_dim = emb_dim
        self.mlp_layers = mlp_layers or [256, 128, 64]
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model: Optional[NCFModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_users: Optional[int] = None
        self.n_movies: Optional[int] = None
        self.user_id_to_idx: Optional[Dict] = None
        self.movie_id_to_idx: Optional[Dict] = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train NCF on the ratings dataframe.

        Drops NaN rows before training to prevent corrupted tensor creation.
        Ratings are normalized to [0, 1] inside RatingsDataset so they match
        the sigmoid output range, which is the primary fix for flat loss.

        Args:
            train_df: DataFrame with columns:
                userId    (int)   - original MovieLens user ID
                movieId   (int)   - original MovieLens movie ID
                user_idx  (int)   - integer index for embedding lookup
                movie_idx (int)   - integer index for embedding lookup
                rating    (float) - user rating in [0.5, 5.0]
        """
        # Drop NaN rows - corrupted index values silently break tensor creation
        train_df = train_df.dropna(subset=["user_idx", "movie_idx", "rating"])
        train_df["user_idx"] = train_df["user_idx"].astype(int)
        train_df["movie_idx"] = train_df["movie_idx"].astype(int)

        self.n_users = int(train_df["user_idx"].max()) + 1
        self.n_movies = int(train_df["movie_idx"].max()) + 1

        # Build lookup maps from original IDs to embedding indices
        self.user_id_to_idx = dict(zip(train_df["userId"], train_df["user_idx"]))
        self.movie_id_to_idx = dict(zip(train_df["movieId"], train_df["movie_idx"]))

        self.model = NCFModel(
            n_users=self.n_users,
            n_movies=self.n_movies,
            emb_dim=self.emb_dim,
            mlp_layers=self.mlp_layers,
            dropout=self.dropout,
        ).to(self.device)

        dataset = RatingsDataset(train_df)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, num_workers=0)
        
        # adding a regularizer with L2 regularization through weight_decay parameter
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)

        # SmoothL1Loss (Huber loss) is better than MSELoss for discrete ratings:
        # - less sensitive to outlier ratings (e.g. troll 1-star ratings)
        # - behaves like MAE far from zero and MSE near zero
        # - results in more stable gradients across the [0.5, 5.0] scale
        criterion = nn.SmoothL1Loss()

        print(f"  Training NCF on {self.device} "
              f"({self.n_users:,} users, {self.n_movies:,} movies)...")

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for user_idx, movie_idx, ratings in loader:
                user_idx = user_idx.to(self.device)
                movie_idx = movie_idx.to(self.device)
                ratings = ratings.to(self.device)

                optimizer.zero_grad()
                preds = self.model(user_idx, movie_idx)

                loss = criterion(preds, ratings)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            elapsed = time.time() - t0
            print(f"    Epoch {epoch + 1}/{self.n_epochs} | "
                  f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        print("  NCF training complete.")

    def predict(self, user_id: int, movie_ids: List[int]) -> Dict[int, float]:
        """
        Predict ratings for a known user across a list of movie IDs.

        Args:
            user_id:   original MovieLens user ID (looked up in user_id_to_idx)
            movie_ids: list of original MovieLens movie IDs to score

        Returns:
            Dict mapping movie_id -> predicted rating in [0.5, 5.0]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .fit() first.")

        user_idx = self.user_id_to_idx.get(user_id)
        if user_idx is None:
            print(f"  Warning: user_id {user_id} not seen during training. "
                  f"Returning default rating of 3.0 for all candidates.")
            return {mid: 3.0 for mid in movie_ids}

        valid_pairs = [
            (mid, self.movie_id_to_idx[mid])
            for mid in movie_ids
            if mid in self.movie_id_to_idx
        ]
        if not valid_pairs:
            print(f"  Warning: none of the {len(movie_ids)} candidate movies "
                  f"were seen during training. Returning empty dict.")
            return {}

        valid_movie_ids, valid_movie_idxs = zip(*valid_pairs)

        self.model.eval()
        with torch.no_grad():
            u_tensor = torch.tensor(
                [user_idx] * len(valid_movie_idxs), dtype=torch.long
            ).to(self.device)
            m_tensor = torch.tensor(
                list(valid_movie_idxs), dtype=torch.long
            ).to(self.device)
            preds = self.model(u_tensor, m_tensor).cpu().numpy()

        return dict(zip(valid_movie_ids, preds.tolist()))

    def predict_new_user(self, ratings: Dict[int, float],
                         candidate_movie_ids: List[int]) -> Dict[int, float]:
        """
        Predict scores for a brand-new user given a few explicit ratings.

        Uses item-based embedding fallback: constructs a pseudo user embedding
        as a rating-weighted average of the rated items' GMF embeddings, then
        scores candidates via dot product with that pseudo embedding.

        All scores are returned in [0.5, 5.0] - consistent with predict().

        Args:
            ratings:              Dict of {movie_id: user_rating} where
                                  user_rating is in [0.5, 5.0]
            candidate_movie_ids:  list of movie IDs to score for this user

        Returns:
            Dict mapping movie_id -> predicted score in [0.5, 5.0]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .fit() first.")

        self.model.eval()
        with torch.no_grad():
            rated_idxs = [
                self.movie_id_to_idx[mid]
                for mid in ratings
                if mid in self.movie_id_to_idx
            ]
            if not rated_idxs:
                print(f"  Warning: none of the rated movies were seen during "
                      f"training. Returning default rating of 3.0 for all candidates.")
                return {mid: 3.0 for mid in candidate_movie_ids}

            # Build pseudo user embedding as weighted average of rated item embeddings
            rated_tensor = torch.tensor(rated_idxs, dtype=torch.long).to(self.device)
            rated_embs = self.model.gmf_item_emb(rated_tensor)

            rating_values = torch.tensor(
                [ratings[mid] for mid in ratings if mid in self.movie_id_to_idx],
                dtype=torch.float32
            ).to(self.device)
            weights = rating_values / rating_values.sum()
            user_emb = (rated_embs * weights.unsqueeze(1)).sum(0, keepdim=True)

            valid_pairs = [
                (mid, self.movie_id_to_idx[mid])
                for mid in candidate_movie_ids
                if mid in self.movie_id_to_idx
            ]
            if not valid_pairs:
                print(f"  Warning: none of the {len(candidate_movie_ids)} candidate "
                      f"movies were seen during training. Returning default of 3.0.")
                return {mid: 3.0 for mid in candidate_movie_ids}

            cand_ids, cand_idxs = zip(*valid_pairs)
            cand_tensor = torch.tensor(list(cand_idxs), dtype=torch.long).to(self.device)
            cand_embs = self.model.gmf_item_emb(cand_tensor)
            scores = torch.matmul(cand_embs, user_emb.squeeze()).cpu().numpy()

            # Normalize dot-product scores to [0.5, 5.0] - same range as predict()
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                scores = RATING_MIN + RATING_RANGE * (scores - min_s) / (max_s - min_s)
            else:
                print(f"  Warning: all candidate scores are identical. "
                      f"Returning default rating of 3.0.")
                scores = np.full_like(scores, 3.0)

        return dict(zip(cand_ids, scores.tolist()))

    def save(self, path: str) -> None:
        """
        Save full NCFRecommender state to disk.

        Saves model weights plus all metadata needed to reconstruct
        the model and perform inference without retraining.

        Args:
            path: file path to save to (should end in .pt)
        """
        state = {
            "model_state_dict": self.model.state_dict(),
            "n_users": self.n_users,
            "n_movies": self.n_movies,
            "emb_dim": self.emb_dim,
            "mlp_layers": self.mlp_layers,
            "dropout": self.dropout,
            "user_id_to_idx": self.user_id_to_idx,
            "movie_id_to_idx": self.movie_id_to_idx,
        }
        torch.save(state, path)
        print(f"  Saved NCFRecommender -> {path}")

    @classmethod
    def load(cls, path: str) -> "NCFRecommender":
        """
        Load NCFRecommender from disk.

        Reconstructs the full model architecture from saved metadata
        and loads trained weights into it.

        Args:
            path: file path to load from (should end in .pt)

        Returns:
            Fully loaded NCFRecommender ready for inference
        """
        state = torch.load(path, map_location="cpu")
        recommender = cls(
            emb_dim=state["emb_dim"],
            mlp_layers=state["mlp_layers"],
            dropout=state["dropout"],
        )
        recommender.n_users = state["n_users"]
        recommender.n_movies = state["n_movies"]
        recommender.user_id_to_idx = state["user_id_to_idx"]
        recommender.movie_id_to_idx = state["movie_id_to_idx"]
        recommender.model = NCFModel(
            n_users=recommender.n_users,
            n_movies=recommender.n_movies,
            emb_dim=recommender.emb_dim,
            mlp_layers=recommender.mlp_layers,
            dropout=recommender.dropout,
        )
        recommender.model.load_state_dict(state["model_state_dict"])
        recommender.model.eval()
        return recommender


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train_all_models() -> None:
    """
    Train all three models on processed training data and save artifacts.

    Saved files:
        models/popularity_baseline.pkl
        models/svd_model.pkl
        models/ncf_model.pt
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("  Loading processed training data...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    print("\n--- Model 1: Popularity Baseline ---")
    popularity = PopularityRecommender()
    popularity.fit(train_df)
    popularity.save(os.path.join(MODELS_DIR, "popularity_baseline.pkl"))

    print("\n--- Model 2: SVD Matrix Factorization ---")
    svd = SVDRecommender(n_factors=100, n_epochs=20)
    svd.fit(train_df)
    rmse = svd.evaluate_rmse(test_df.sample(min(10000, len(test_df))))
    print(f"  SVD Test RMSE: {rmse:.4f}")
    svd.save(os.path.join(MODELS_DIR, "svd_model.pkl"))

    print("\n--- Model 3: Neural Collaborative Filtering ---")
    ncf = NCFRecommender(emb_dim=64, mlp_layers=[256, 128, 64],
                         dropout=0.2, lr=1e-3, batch_size=1024, n_epochs=10)
    ncf.fit(train_df)
    ncf.save(os.path.join(MODELS_DIR, "ncf_model.pt"))

    print("\n  All models trained and saved to models/")


if __name__ == "__main__":
    train_all_models()
