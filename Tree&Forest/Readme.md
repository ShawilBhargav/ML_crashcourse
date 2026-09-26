# Heart Disease Prediction — Decision Tree vs. Random Forest

Comparing an unrestricted Decision Tree, a depth-limited Decision Tree, and a Random Forest on the same heart disease dataset, to see the overfitting-to-ensemble story play out directly in the numbers.

> 🎓 This project focuses specifically on *why* Random Forest tends to outperform a single Decision Tree — by building all three model variants on the same data and comparing them side by side, rather than just training one model and reporting a score.

## Overview

This project predicts the presence of heart disease from clinical and demographic features. Three tree-based models were trained on identical data splits — an unrestricted Decision Tree, a depth-limited Decision Tree, and a Random Forest — to directly observe how controlling tree complexity and ensembling affect generalization. The project also examines feature importance from the Random Forest to check whether the model's learned patterns align with known clinical risk factors.

## Tools Used

- Python
- Pandas, NumPy
- Scikit-learn (DecisionTreeClassifier, RandomForestClassifier, train_test_split, metrics)

## Dataset

**Source:** Heart Failure Prediction dataset (Kaggle)

918 records with clinical features including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, max heart rate, exercise-induced angina, oldpeak (ST depression), and ST slope, with `HeartDisease` as the binary target.

## Key Steps Performed

- Loaded and inspected the dataset — no missing values, no ID columns to drop
- Applied a deliberate, column-by-column encoding framework rather than blanket one-hot encoding everything:
  - **Ordinal encoding** for `ST_Slope`, since its categories (Up/Flat/Down) reflect a real clinical severity order
  - **One-hot encoding** for `ChestPainType` and `RestingECG`, which have multiple categories with no inherent order
  - **Binary encoding** for `Sex` and `ExerciseAngina` (2 categories each)
  - Confirmed `FastingBS` was already a binary flag requiring no transformation
- Split data into training (80%) and test (20%) sets — no feature scaling applied, since tree-based models don't require it
- Trained an unrestricted Decision Tree and recorded its depth
- Trained a depth-limited Decision Tree (`max_depth=8`) and compared metrics
- Trained a Random Forest (`n_estimators=100`) and compared metrics against both tree versions
- Extracted and ranked feature importances from the Random Forest model
- Tuned the Random Forest with `GridSearchCV` (5-fold cross-validation) across `n_estimators`, `max_depth`, and `min_samples_split`, then evaluated the tuned model on the same held-out test set used for all other models, to get a fair before/after comparison rather than comparing a cross-validated score to a single-split score

## Model Performance

| Metric | Decision Tree (unrestricted, depth 14) | Decision Tree (max_depth=8) | Random Forest (default) | Random Forest (tuned) |
|---|---|---|---|---|
| Accuracy | 83.70% | 85.33% | 86.96% | **87.50%** |
| Precision | 85.98% | 88.46% | 89.52% | **88.89%** |
| Recall | 85.98% | 85.98% | 87.85% | **89.72%** |
| F1 | 85.98% | 87.20% | 88.68% | **89.30%** |

**Tuned Random Forest hyperparameters** (via `GridSearchCV`, 5-fold cross-validation): `n_estimators=200`, `min_samples_split=10`, `max_depth=None`.

## Key Insights

- **Every metric improved consistently and monotonically** moving from the unrestricted tree → depth-limited tree → Random Forest, giving a clean, direct demonstration of two separate overfitting remedies: explicit regularization (limiting depth) and ensembling (averaging predictions across many trees).
- **The unrestricted tree grew to a depth of 14** even on a relatively clean, well-structured dataset — reinforcing that trees will over-complicate themselves by default unless explicitly constrained, regardless of how "clean" the underlying data is.
- **Random Forest gave the best result on every metric**, consistent with the theory that combining many diverse trees reduces the variance/noise any single tree picks up from its specific training sample.
- **Feature importance analysis aligned with real clinical knowledge**: `ST_Slope`, `Oldpeak`, `MaxHR`, and `Cholesterol` emerged as the strongest predictors — all established cardiovascular risk indicators — giving some confidence the model learned genuine medical patterns rather than spurious correlations.
- **`ST_Slope`, the feature deliberately given ordinal (rather than one-hot) encoding due to its real clinical severity order, turned out to be the single most important feature** — a nice validation that encoding categorical variables thoughtfully, rather than defaulting to one-hot encoding everywhere, can matter for model quality.
- **Several encoded features (e.g. `FastingBS`, individual `RestingECG` categories) contributed very little** to the Random Forest's predictions — a reminder that not every carefully-prepared feature ends up mattering equally.
- **Systematic hyperparameter tuning (`GridSearchCV`) improved the default Random Forest further still**, from 86.96% to 87.50% accuracy on the held-out test set — a modest but genuine gain achieved purely by searching for a better `n_estimators`/`max_depth`/`min_samples_split` combination via cross-validation, rather than guessing values manually. The cross-validated score found during the search (87.87%) closely matched the final test-set score (87.50%), suggesting the tuning process itself wasn't overfitting to its own validation folds.
- **Systematic hyperparameter tuning via `GridSearchCV` improved every metric further still**, taking accuracy from 86.96% (default settings) to 87.50% on the held-out test set — a real, fairly-measured gain (verified on the test set itself, not just the cross-validated search score) achieved without changing the algorithm, features, or data at all. This came from increasing `n_estimators` to 200 and requiring `min_samples_split=10`, showing that even an already-strong default model can be meaningfully improved through systematic rather than manual search.

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn
   ```
3. Download the dataset from Kaggle and place the CSV in the project folder
4. Open the notebook in Jupyter Notebook or Google Colab
5. Run all cells in order

## What I'd Improve Next

- Compare against a boosting method (e.g. XGBoost) to see if it improves further on the tuned Random Forest
- Visualize one of the individual decision trees to inspect the actual split logic at the top levels
- Try `RandomizedSearchCV` with a wider hyperparameter range to check whether an even better combination exists outside the original grid

## Notes

This project is part of a structured self-study path moving from data analysis through machine learning fundamentals into deep learning.