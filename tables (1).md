# Results for the revised manuscript

```
{
  "n_records": 200000,
  "n_features_raw": 31,
  "n_fraud": 2206,
  "fraud_rate_pct": 1.1,
  "selected_features": [
    "proposed_credit_limit",
    "credit_risk_score",
    "customer_age",
    "keep_alive_session",
    "income",
    "housing_status",
    "date_of_birth_distinct_emails_4w",
    "phone_home_valid",
    "current_address_months_count",
    "name_email_similarity",
    "has_other_cards",
    "device_os",
    "device_distinct_emails_8w",
    "email_is_free",
    "prev_address_months_count",
    "foreign_request"
  ],
  "cost_optimal_threshold": 0.7,
  "latency_ms_median": 78.22,
  "latency_ms_p95": 99.74
}
```

## Table 3. Cross-validated comparison

| Model                    | Accuracy      | Balanced acc.   | Precision     | Recall       | F1           | ROC-AUC      | PR-AUC       | MCC          |   Train time/fold (s) |
|:-------------------------|:--------------|:----------------|:--------------|:-------------|:-------------|:-------------|:-------------|:-------------|----------------------:|
| Logistic Regression      | 80.22 ± 0.19  | 79.26 ± 0.66    | 4.23 ± 0.07   | 78.29 ± 1.37 | 8.03 ± 0.13  | 86.83 ± 0.81 | 11.58 ± 1.23 | 15.17 ± 0.33 |                3.8000 |
| Gaussian NB              | 56.07 ± 21.13 | 71.49 ± 7.91    | 2.38 ± 0.61   | 87.26 ± 6.13 | 4.63 ± 1.16  | 82.70 ± 1.58 | 6.88 ± 0.99  | 9.57 ± 3.08  |                1.7000 |
| Random Forest            | 98.90 ± 0.00  | 50.02 ± 0.05    | 10.00 ± 22.36 | 0.05 ± 0.10  | 0.09 ± 0.20  | 83.61 ± 0.89 | 9.55 ± 0.68  | 0.64 ± 1.49  |               75.7000 |
| XGBoost                  | 92.40 ± 0.21  | 72.47 ± 0.69    | 7.51 ± 0.19   | 52.09 ± 1.50 | 13.13 ± 0.30 | 84.91 ± 0.81 | 10.17 ± 0.79 | 17.66 ± 0.45 |                9.8000 |
| LightGBM                 | 90.36 ± 0.72  | 71.46 ± 0.84    | 5.96 ± 0.50   | 52.13 ± 1.38 | 10.70 ± 0.83 | 83.02 ± 0.64 | 8.39 ± 0.78  | 15.19 ± 0.99 |                7.5000 |
| Soft voting (GNB+RF+XGB) | 92.76 ± 0.53  | 72.71 ± 0.65    | 7.92 ± 0.44   | 52.22 ± 1.63 | 13.75 ± 0.66 | 85.44 ± 1.15 | 10.61 ± 0.62 | 18.27 ± 0.60 |               78.2000 |
| Proposed stack           | 94.44 ± 0.26  | 69.17 ± 0.91    | 8.83 ± 0.43   | 43.34 ± 1.92 | 14.67 ± 0.64 | 84.80 ± 0.89 | 9.73 ± 1.23  | 17.70 ± 0.79 |              391.7000 |

## Table 4. Ablation

| Model                                          | Accuracy     | Balanced acc.   | Precision     | Recall       | F1           | ROC-AUC      | PR-AUC       | MCC          |   Train time/fold (s) |
|:-----------------------------------------------|:-------------|:----------------|:--------------|:-------------|:-------------|:-------------|:-------------|:-------------|----------------------:|
| Full stack (GNB+RF+XGB, passthrough, XGB meta) | 94.44 ± 0.26 | 69.17 ± 0.91    | 8.83 ± 0.43   | 43.34 ± 1.92 | 14.67 ± 0.64 | 84.80 ± 0.89 | 9.73 ± 1.23  | 17.70 ± 0.79 |              398.3000 |
| Without passthrough                            | 84.94 ± 0.39 | 75.58 ± 1.72    | 4.73 ± 0.28   | 66.00 ± 3.34 | 8.82 ± 0.52  | 83.33 ± 1.24 | 10.27 ± 0.57 | 14.80 ± 1.05 |              385.8000 |
| Without GNB                                    | 94.31 ± 0.23 | 69.13 ± 0.31    | 8.65 ± 0.31   | 43.38 ± 0.75 | 14.42 ± 0.42 | 84.69 ± 1.00 | 9.68 ± 0.83  | 17.48 ± 0.37 |              396.7000 |
| Without RF                                     | 94.35 ± 0.28 | 69.02 ± 1.06    | 8.67 ± 0.60   | 43.11 ± 2.05 | 14.43 ± 0.93 | 84.72 ± 0.58 | 10.58 ± 0.97 | 17.45 ± 1.12 |               57.2000 |
| Without XGB base                               | 93.47 ± 0.29 | 70.50 ± 0.67    | 8.03 ± 0.30   | 47.01 ± 1.50 | 13.71 ± 0.45 | 84.33 ± 0.99 | 10.09 ± 1.35 | 17.42 ± 0.52 |              342.9000 |
| LR meta-learner                                | 98.90 ± 0.01 | 50.18 ± 0.17    | 40.00 ± 30.28 | 0.36 ± 0.34  | 0.72 ± 0.68  | 87.07 ± 0.83 | 12.16 ± 1.21 | 3.64 ± 3.04  |              376.1000 |

## Table 5. Leakage check

| Model                             | Accuracy     | Balanced acc.   | Precision   | Recall       | F1           | ROC-AUC      | PR-AUC      | MCC          |   Train time/fold (s) |
|:----------------------------------|:-------------|:----------------|:------------|:-------------|:-------------|:-------------|:------------|:-------------|----------------------:|
| All features                      | 94.44 ± 0.26 | 69.17 ± 0.91    | 8.83 ± 0.43 | 43.34 ± 1.92 | 14.67 ± 0.64 | 84.80 ± 0.89 | 9.73 ± 1.23 | 17.70 ± 0.79 |              391.7000 |
| Without engineered price features | 94.44 ± 0.26 | 69.17 ± 0.91    | 8.83 ± 0.43 | 43.34 ± 1.92 | 14.67 ± 0.64 | 84.80 ± 0.89 | 9.73 ± 1.23 | 17.70 ± 0.79 |              407.3000 |

## Table 6. Significance tests

| Comparison                | Test            |   Proposed right / other wrong |   Other right / proposed wrong |   p-value |
|:--------------------------|:----------------|-------------------------------:|-------------------------------:|----------:|
| Proposed vs XGBoost       | McNemar (exact) |                           1581 |                            496 |    0.0000 |
| Proposed vs Random Forest | McNemar (exact) |                            197 |                           2034 |    0.0000 |

## Table 7. Manual-review band

| Review band   |   Sent to manual review (%) |   Auto-decided error rate (%) |   Fraud missed by auto-accept |
|:--------------|----------------------------:|------------------------------:|------------------------------:|
| [0.2, 0.8)    |                     13.7625 |                        1.7887 |                           128 |
| [0.3, 0.7)    |                      8.2900 |                        2.8159 |                           167 |
| [0.4, 0.6)    |                      3.9050 |                        4.0949 |                           200 |

## Holdout confusion matrix

|   Threshold |       TP |        FP |       FN |         TN |   Accuracy |   Balanced acc. |   Precision |   Recall |      F1 |   ROC-AUC |   PR-AUC |     MCC |
|------------:|---------:|----------:|---------:|-----------:|-----------:|----------------:|------------:|---------:|--------:|----------:|---------:|--------:|
|      0.5000 | 198.0000 | 2034.0000 | 243.0000 | 37525.0000 |    94.3075 |         69.8781 |      8.8710 |  44.8980 | 14.8148 |   84.4520 |   8.8477 | 18.0858 |
|      0.7000 | 124.0000 |  866.0000 | 317.0000 | 38693.0000 |    97.0425 |         62.9644 |     12.5253 |  28.1179 | 17.3305 |   84.4520 |   8.8477 | 17.4268 |

## Permutation importance (top 15)

| feature                          |   importance_mean |   importance_sd |
|:---------------------------------|------------------:|----------------:|
| housing_status                   |            0.0317 |          0.0036 |
| device_os                        |            0.0269 |          0.0043 |
| phone_home_valid                 |            0.0133 |          0.0027 |
| has_other_cards                  |            0.0127 |          0.0035 |
| credit_risk_score                |            0.0096 |          0.0029 |
| date_of_birth_distinct_emails_4w |            0.0088 |          0.0025 |
| prev_address_months_count        |            0.0067 |          0.0030 |
| keep_alive_session               |            0.0062 |          0.0037 |
| income                           |            0.0045 |          0.0023 |
| name_email_similarity            |            0.0033 |          0.0055 |
| customer_age                     |            0.0029 |          0.0029 |
| current_address_months_count     |            0.0024 |          0.0044 |
| email_is_free                    |            0.0022 |          0.0034 |
| foreign_request                  |            0.0007 |          0.0006 |
| device_distinct_emails_8w        |            0.0006 |          0.0008 |

