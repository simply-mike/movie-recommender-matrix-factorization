# Source code

The current version is notebook-first.

Recommended future split:

```text
src/
├── data.py          # MovieLens loading and temporal split
├── baselines.py     # Most Popular, User-CF, Item-CF
├── model.py         # matrix factorization model
├── train.py         # training loop
├── metrics.py       # RMSE and ranking metrics
└── utils.py
```
