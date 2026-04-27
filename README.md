# Movie Recommender with Matrix Factorization

A recommender system project based on **MovieLens 100k**. The project compares simple collaborative filtering baselines with a neural matrix factorization model trained with PyTorch embeddings.

The focus is not only rating prediction, but also **top-k recommendation quality**.

## Problem

Given historical user-movie ratings, recommend unseen movies for each user.

Two evaluation views are used:

1. **Rating prediction**
   - RMSE

2. **Top-k recommendation**
   - Precision@10
   - Recall@10
   - MAP@10
   - NDCG@10
   - Coverage

For recommender systems, ranking metrics are usually more aligned with the product task than RMSE, because the user ultimately sees a short ranked list of recommendations.

## Dataset

Dataset: **MovieLens 100k**

The dataset contains user-item ratings with timestamps.

The data is split per user using a temporal protocol:

```text
train: all interactions except the last two
validation: second-to-last interaction
test: last interaction
```

Resulting split:

| Split | Interactions |
|---|---:|
| Train | 98,114 |
| Validation | 943 |
| Test | 943 |

This avoids the main leakage problem of random splits: using future interactions to predict earlier user behavior.

## Models

### Baselines

| Model | Description |
|---|---|
| Most Popular | recommends globally popular items |
| User-based CF | neighborhood-based collaborative filtering over users |
| Item-based CF | neighborhood-based collaborative filtering over items |

### Matrix factorization

The main model uses user and item embeddings:

```text
score(user, item) = dot(user_embedding, item_embedding) + optional biases
```

Tested configurations include:

- latent dimensions: 16, 32, 64
- L2 regularization: `1e-4`, `1e-3`
- optimizers: Adam, SGD
- with and without bias terms

## Results

### Validation baselines

| Model | Val RMSE | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Most Popular | 1.0641 | 0.0095 | 0.0953 | 0.0303 | 0.0449 | 0.0482 |
| User-CF | 1.0669 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0184 |
| Item-CF | 1.1108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1754 |

### Matrix factorization experiments

| Experiment | Best val epoch | Best val RMSE | Best val NDCG@10 | Test RMSE | Test Precision@10 | Test Recall@10 | Test MAP@10 | Test NDCG@10 | Test Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mf_f16_reg1e-4_adam_bias0` | 3 | 1.2265 | **0.0522** | 1.2376 | 0.0109 | 0.1091 | 0.0358 | **0.0528** | 0.0422 |
| `mf_f64_reg1e-4_adam_bias1` | 2 | 1.3886 | 0.0323 | 1.3800 | 0.0047 | 0.0473 | 0.0151 | 0.0225 | 0.0612 |
| `mf_f32_reg1e-3_adam_bias1` | 3 | 1.1981 | 0.0255 | 1.2362 | 0.0049 | 0.0494 | 0.0169 | 0.0242 | 0.0553 |
| `mf_f32_reg1e-4_adam_bias1` | 2 | 1.6090 | 0.0238 | 1.6069 | 0.0039 | 0.0391 | 0.0108 | 0.0172 | 0.0464 |
| `mf_f32_reg1e-4_sgd_bias1` | 1 | 2.6629 | 0.0006 | 2.6959 | 0.0004 | 0.0041 | 0.0005 | 0.0012 | 0.0107 |

## Interpretation

The best ranking quality was achieved by the smallest matrix factorization model:

```text
16 latent factors, Adam optimizer, L2 regularization = 1e-4, no bias terms
```

The best matrix factorization model achieved Recall@10 = 0.1091 and NDCG@10 = 0.0528 on the test split. Since the evaluation uses a temporal leave-last-item-out setup, Recall@10 can be interpreted as the fraction of users for whom the held-out next item appeared in the top-10 recommendations.

The model captures part of the user preference structure, but the low coverage value indicates that recommendations are concentrated on a relatively small subset of the item catalog.

## Example recommendations

For one fixed test user, the trained model recommended:

| Rank | Movie |
|---:|---|
| 1 | Titanic (1997) |
| 2 | Star Wars (1977) |
| 3 | Full Monty, The (1997) |
| 4 | Godfather, The (1972) |
| 5 | Fargo (1996) |
| 6 | English Patient, The (1996) |
| 7 | Trainspotting (1996) |
| 8 | Apt Pupil (1998) |
| 9 | Casablanca (1942) |
| 10 | Silence of the Lambs, The (1991) |

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/matrix_factorization_recommender.ipynb
```

Start TensorBoard:

```bash
tensorboard --logdir runs/task8
```