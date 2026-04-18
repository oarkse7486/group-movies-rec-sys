"""
train_ncf.py
------------
Standalone script to train only the NCF (Neural Collaborative Filtering)
model without rerunning the full setup pipeline.

Loads preprocessed training data from data/processed/train.csv, trains
the NCF model for the specified number of epochs, and saves the trained
model artifact to models/ncf_model.pt.

All training output (epoch loss, timing) is printed to the terminal AND
saved simultaneously to data/outputs/ncf_training_log.txt for later review.

Usage:
    python3 train_ncf.py

To adjust training, change n_epochs before running:
    - n_epochs=3  for a quick test to confirm loss is dropping
    - n_epochs=20 for a full overnight training run
"""

import sys
import os
from scripts.model import NCFRecommender
import pandas as pd

os.makedirs("data/outputs", exist_ok=True)


class Tee:
    """
    Writes output to both terminal and a log file at the same time.
    Stores a reference to the original stdout to avoid infinite recursion.
    """
    def __init__(self, log_path: str):
        self._terminal = sys.__stdout__   # reference to real stdout, not sys.stdout
        self._log = open(log_path, 'w')

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()


tee = Tee('data/outputs/ncf_training_log.txt')
sys.stdout = tee

train_df = pd.read_csv('data/processed/train.csv')

ncf = NCFRecommender(
    emb_dim=64,
    mlp_layers=[256, 128, 64],
    dropout=0.2,
    lr=1e-3,
    batch_size=1024,
    n_epochs=20       # change to 20 for full training run
)

ncf.fit(train_df)
ncf.save('models/ncf_model.pt')

tee.close()
sys.stdout = sys.__stdout__
