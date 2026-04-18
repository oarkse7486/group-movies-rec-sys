"""
scripts/make_dataset.py
-----------------------
Downloads and validates the MovieLens 25M dataset from GroupLens.
Saves raw files to data/raw/.

Usage:
    python scripts/make_dataset.py
"""

import os
import zipfile
import requests
from tqdm import tqdm

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
RAW_DIR = "data/raw"
ZIP_PATH = os.path.join(RAW_DIR, "ml-25m.zip")

REQUIRED_FILES = [
    "ml-25m/ratings.csv",
    "ml-25m/movies.csv",
    "ml-25m/tags.csv",
    "ml-25m/links.csv",
]


def _files_already_exist() -> bool:
    """Check if all required MovieLens files are already present."""
    for fname in REQUIRED_FILES:
        if not os.path.exists(os.path.join(RAW_DIR, fname)):
            return False
    return True


def _download_zip() -> None:
    """Stream-download the MovieLens zip with a progress bar."""
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"  Downloading from {MOVIELENS_URL}")
    response = requests.get(MOVIELENS_URL, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(ZIP_PATH, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="  ml-25m.zip"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def _extract_zip() -> None:
    """Extract the downloaded zip file to data/raw/."""
    print("  Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)
    os.remove(ZIP_PATH)
    print("  Extraction complete.")


def download_movielens() -> None:
    """
    Download MovieLens 25M if not already present.

    Skips download if all required files already exist in data/raw/.
    """
    if _files_already_exist():
        print("  MovieLens 25M already downloaded. Skipping.")
        return

    _download_zip()
    _extract_zip()
    print("  MovieLens 25M ready in data/raw/ml-25m/")


if __name__ == "__main__":
    download_movielens()
