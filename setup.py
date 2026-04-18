"""
setup.py
--------
Orchestrates the full Group Rec setup pipeline:
  1. Download MovieLens 25M dataset
  2. Preprocess and split data
  3. Train all three models (baseline, SVD, NCF)
  4. Save model artifacts to models/

Run with: python3 setup.py
"""

from scripts.make_dataset import download_movielens
from scripts.build_features import run_preprocessing_pipeline
from scripts.model import train_all_models


def main():
    """Run the full setup pipeline."""
    print("=" * 60)
    print("  Group Rec - Setup Pipeline")
    print("=" * 60)

    print("\n[1/3] Downloading MovieLens dataset...")
    download_movielens()

    print("\n[2/3] Running preprocessing pipeline...")
    run_preprocessing_pipeline()

    print("\n[3/3] Training all models...")
    train_all_models()

    print("\n" + "=" * 60)
    print("  Setup complete. Run `make serve` to start the API.")
    print("=" * 60)


if __name__ == "__main__":
    main()