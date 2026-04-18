# Group Rec

Group Rec is a group-aware movie recommendation system that solves a genuinely hard problem: what should a group of people with different tastes watch together?

Unlike traditional recommenders that optimize for a single user, Group Rec builds individual taste profiles for each member and aggregates them using three strategies - Least Misery, Average Satisfaction, and Fairness-Aware - each with different tradeoffs between accuracy and fairness.

---

## Live App

https://group-rec.vercel.app (update after deployment)

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
python setup.py
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

## Group Aggregation Strategies

| Strategy | Logic | Best For |
|---|---|---|
| Least Misery | Min of individual scores | No one hates it |
| Average Satisfaction | Mean of individual scores | Most overall happiness |
| Fairness-Aware | alpha x avg + (1-alpha) x least misery | Balanced tradeoff |

Implemented in scripts/group_aggregation.py.

---

## Experiment

Research question: How does group size (2-10 members) affect the accuracy-fairness tradeoff across aggregation strategies?

Results are logged to data/outputs/experiment_results.csv and visualized in notebooks/03_group_aggregation_analysis.ipynb.

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
