# Group Rec

Group Rec is a group-aware movie recommendation system that solves a genuinely hard problem: what should a group of people with different tastes watch together?

Unlike traditional recommenders that optimize for a single user, Group Rec builds individual taste profiles for each member and aggregates them using three strategies - Least Misery, Average Satisfaction, and Fairness-Aware - each with different tradeoffs between accuracy and fairness.

---


## Features

- Supports groups of 2-10 members
- Three group aggregation strategies with plain-English explanations
- Per-member satisfaction visualization for every recommendation
- Group-level fairness dashboard
- Backed by three RS models: popularity baseline, SVD, and Neural CF

---

## Project Structure

```
group-rec/
|-- README.md
|-- requirements.txt
|-- setup.py                    <- downloads data, trains and saves all models
|-- main.py                     <- FastAPI inference server
|-- Makefile                    <- make setup | make train | make serve
|-- evaluate.py                 <- computes RMSE, MAE, NDCG@10 for all models
|-- experiment.py               <- sensitivity analysis: group size vs accuracy-fairness tradeoff
|-- train_ncf.py                <- standalone NCF training script with logging
|-- scripts/
|   |-- make_dataset.py         <- downloads and validates MovieLens 25M
|   |-- build_features.py       <- preprocessing pipeline and train/test splits
|   |-- model.py                <- trains all 3 models, saves artifacts
|   `-- group_aggregation.py    <- least misery, avg satisfaction, fairness-aware
|-- models/                     <- saved model artifacts (.pkl, .pt)
|-- data/
|   |-- raw/                    <- MovieLens 25M source files
|   |-- processed/              <- user-item matrix, train/test splits
|   `-- outputs/                <- experiment results, evaluation logs
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_model_experiments.ipynb
|   `-- 03_group_aggregation_analysis.ipynb
|-- frontend/                   <- React application
`-- .gitignore
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data, preprocess, and train all models
```bash
make setup
```
Or run manually:
```bash
python3 setup.py
```

### 3. Start the API server
```bash
make serve
```
Or:
```bash
uvicorn main:app --reload --port 8000
```

### 4. Run the frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Models

| Model | Type | Location |
|---|---|---|
| Popularity Baseline | Naive (most-rated films) | scripts/model.py -> PopularityRecommender |
| SVD | Classical Matrix Factorization | scripts/model.py -> SVDRecommender |
| Neural CF (NCF) | Deep Learning (PyTorch) | scripts/model.py -> NCFRecommender |

All three models are saved to the models/ directory after running `make setup`.

---

## Evaluation Results

Evaluated on the held-out test set (4,866,448 ratings, random_state=42).

| Model | RMSE | MAE | NDCG@10 |
|---|---|---|---|
| Popularity Baseline | 1.5252 | 1.0591 | 0.7664 |
| SVD | 0.7856 | 0.5893 | 0.8165 |
| NCF | 0.8026 | 0.5960 | 0.8168 |

Run evaluation:
```bash
python3 evaluate.py
```

---

## Group Aggregation Strategies

| Strategy | Logic | Best For |
|---|---|---|
| Least Misery | Min of individual scores | No one hates it |
| Average Satisfaction | Mean of individual scores | Most overall happiness |
| Fairness-Aware | alpha x avg + (1-alpha) x least misery | Balanced tradeoff |

Implemented in scripts/group_aggregation.py.

---

## Experiment

Sensitivity analysis: how does group size (2-10 members) affect the accuracy-fairness tradeoff across aggregation strategies?

Selected results (random_state=42, N=100 groups per size):

| Group Size | Strategy | Avg Satisfaction | Fairness Score |
|---|---|---|---|
| 2 | Least Misery | 4.4811 | 0.9601 |
| 2 | Average Satisfaction | 4.4986 | 0.9406 |
| 2 | Fairness-Aware | 4.4950 | 0.9499 |
| 5 | Least Misery | 4.4039 | 0.9258 |
| 5 | Average Satisfaction | 4.4350 | 0.9089 |
| 5 | Fairness-Aware | 4.4257 | 0.9192 |
| 10 | Least Misery | 4.3573 | 0.9005 |
| 10 | Average Satisfaction | 4.4044 | 0.8863 |
| 10 | Fairness-Aware | 4.3861 | 0.8971 |

Full results saved to data/outputs/experiment_results.csv. Run:
```bash
python3 experiment.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /recommend | Get group recommendations |
| GET | /movies/popular | Fetch popular movies for rating UI |
| POST | /users/profile | Build a taste profile from ratings |
| GET | /health | Health check |

---

## Tech Stack

- Backend: FastAPI, PyTorch, Surprise, Pandas
- Frontend: React, Tailwind CSS
- Deployment: Render (API) + Vercel (frontend)
- Data: MovieLens 25M + TMDB API

---

## Dataset

MovieLens 25M (https://grouplens.org/datasets/movielens/25m/) - 25 million ratings, 62,000 movies, 162,000 users. Downloaded automatically via `make setup`.

---

## Ethics Statement

See report for full ethics discussion. Key considerations: popularity bias systematically disadvantages minority-taste users in groups; fairness-aware aggregation partially mitigates but does not eliminate this. No personal data is stored beyond the current session.
