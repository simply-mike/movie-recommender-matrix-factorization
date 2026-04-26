# Results: Movie Recommender with Matrix Factorization

## Validation baselines

| Model | Val RMSE | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Most Popular | 1.0641 | 0.0095 | 0.0953 | 0.0303 | 0.0449 | 0.0482 |
| User-CF | 1.0669 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0184 |
| Item-CF | 1.1108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1754 |

## Matrix factorization

| Experiment | Best val epoch | Best val RMSE | Best val NDCG@10 | Test RMSE | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mf_f16_reg1e-4_adam_bias0` | 3 | 1.2265 | 0.0522 | 1.2376 | 0.0109 | 0.1091 | 0.0358 | 0.0528 | 0.0422 |
| `mf_f64_reg1e-4_adam_bias1` | 2 | 1.3886 | 0.0323 | 1.3800 | 0.0047 | 0.0473 | 0.0151 | 0.0225 | 0.0612 |
| `mf_f32_reg1e-3_adam_bias1` | 3 | 1.1981 | 0.0255 | 1.2362 | 0.0049 | 0.0494 | 0.0169 | 0.0242 | 0.0553 |
| `mf_f32_reg1e-4_adam_bias1` | 2 | 1.6090 | 0.0238 | 1.6069 | 0.0039 | 0.0391 | 0.0108 | 0.0172 | 0.0464 |
| `mf_f32_reg1e-4_sgd_bias1` | 1 | 2.6629 | 0.0006 | 2.6959 | 0.0004 | 0.0041 | 0.0005 | 0.0012 | 0.0107 |

## Conclusion

The best model by NDCG@10 was `mf_f16_reg1e-4_adam_bias0`.

Key observations:

- Temporal validation is stricter and more realistic than random splitting.
- The best model is compact: 16 latent factors were enough for this dataset.
- Larger factor sizes did not improve top-k ranking.
- RMSE and ranking metrics disagree, so model selection should follow the final recommendation objective rather than only rating reconstruction error.
