# Model Comparison — Phase 6 Full Development

| model                 |   accuracy |   precision |   recall |   f1_score |   roc_auc |   pr_auc |
|:----------------------|-----------:|------------:|---------:|-----------:|----------:|---------:|
| Random Forest (tuned) |   0.999989 |    1        | 0.996348 |   0.998171 |  0.998461 | 0.996924 |
| XGBoost (tuned)       |   0.999462 |    0.848626 | 0.996348 |   0.916573 |  0.999092 | 0.993584 |
| Isolation Forest      |   0.994333 |    0.025966 | 0.024954 |   0.02545  |  0.890117 | 0.029554 |
| Logistic Regression   |   0.94671  |    0.048715 | 0.916007 |   0.09251  |  0.97983  | 0.60412  |