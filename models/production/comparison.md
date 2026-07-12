# Model Performance Comparison

Below is the performance comparison showing both CV and test metrics side-by-side:

| Run Name                     | Model Type          |   CV ROC-AUC (Mean) |   Test ROC-AUC |   Accuracy |   Precision |   Recall |   F1-Score |
|:-----------------------------|:--------------------|--------------------:|---------------:|-----------:|------------:|---------:|-----------:|
| Logistic_Regression_Baseline | logistic_regression |              0.9074 |         0.9545 |     0.8689 |      0.8333 |   0.8929 |     0.8621 |
| Logistic_Regression_Baseline | logistic_regression |              0.9074 |         0.9545 |     0.8689 |      0.8333 |   0.8929 |     0.8621 |
| Logistic_Regression_Baseline | logistic_regression |              0.9074 |         0.9545 |     0.8689 |      0.8333 |   0.8929 |     0.8621 |
| XGBoost_Ensemble             | xgboost             |              0.8897 |         0.9459 |     0.9016 |      0.8667 |   0.9286 |     0.8966 |
| XGBoost_Ensemble             | xgboost             |              0.8897 |         0.9459 |     0.9016 |      0.8667 |   0.9286 |     0.8966 |

Best model selected is **Logistic_Regression_Baseline** based on CV ROC-AUC (with F1-score tiebreaker if difference < 0.01).
