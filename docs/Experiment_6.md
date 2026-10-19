#Experiment 6: Statistical Analysis, Model Validation, and Research Interpretation

## 1. Objective
To perform inferential statistical tests on fitness tracking data, benchmark four regression models, conduct 5-fold cross-validation, and evaluate prototype prediction accuracy.

## 2. Statistical Findings
- **Pearson Correlation (Duration vs. Calories)**: r = 0.9081 (p < 0.001), proving strong linear dependency on duration.
- **Welch Two-Sample t-Test (Gender)**: t = -4.7814 (p < 0.001), significant calorie expenditure variation attributed to average body mass differences.
- **One-Way ANOVA (Workout Types)**: F = 0.9490 (p = 0.4162), indicating similar calorie burns across workout categories when duration is held constant.
- **Chi-Square Independence Test**: chi² = 5.7903 (p = 0.4471), showing user experience levels are independent of workout type selection.

## 3. Benchmark Results
| Model | Test MAE | Test RMSE | Test R² | 5-Fold CV R² |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **30.27** | **40.57** | **0.9803** | **0.9783** |
| **Random Forest** | 36.74 | 49.01 | 0.9712 | 0.9750 |
| **SVR** | 42.15 | 56.88 | 0.9614 | 0.9632 |
| **Decision Tree** | 69.71 | 89.14 | 0.9048 | 0.9064 |

## 4. Execution & Testing
- Automated unit test suites configured with `pytest` in `tests/test_prediction.py`.
- Validation checks test dataset bounds, schema alignment, and output range verification ($100 \le \text{Calories} \le 2000$).
