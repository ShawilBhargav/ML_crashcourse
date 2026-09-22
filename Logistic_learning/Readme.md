# Bank Customer Churn Prediction — Logistic Regression

Predicting whether a bank customer will churn (leave the bank), with a deliberate focus on catching and correcting a real data leakage bug — and what that taught me about trusting suspiciously good results.

> 🎓 This is my third ML project, and the first one where I ran into a genuine, classic ML mistake (data leakage) rather than a clean run. I'm documenting that mistake and the fix here rather than hiding it, since diagnosing it was the most valuable part of the project.

## Overview

This project builds a Logistic Regression model to predict customer churn using demographic, account, and product-usage features. The initial model returned a suspiciously perfect 99.9% accuracy — investigating why led to identifying and removing a leaked feature, after which a much more realistic (and much more interesting) model emerged, exposing a real class-imbalance problem worth solving for.

## Tools Used

- Python
- Pandas, NumPy
- Scikit-learn (LogisticRegression, StandardScaler, train_test_split, metrics)

## Dataset

**Source:** Bank Customer Churn dataset (Kaggle)

~10,000 customer records with features including credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, activity status, estimated salary, and the target `Exited` (1 = churned, 0 = stayed).

## Key Steps Performed

- Loaded and inspected the dataset; dropped non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- Encoded categorical features (`Geography`, `Gender`) using one-hot encoding
- Split data into training (80%) and test (20%) sets
- Scaled features with `StandardScaler`, fit on training data only, to avoid data leakage through preprocessing
- Trained a Logistic Regression model
- **Caught a data leakage bug**: an initial run using all available columns returned 99.9% accuracy — investigated and found the `Complain` column was almost perfectly correlated with the target, since it was effectively recorded as a result of the churn event rather than as a genuine, independently-knowable predictor
- Removed the leaked feature and retrained, producing an honest, realistic accuracy of ~81%
- Diagnosed a large gap between accuracy (81%) and recall (20%), tracing it to class imbalance in the dataset (~80% of customers stayed)
- Reasoned through which metric a bank should actually prioritize (recall, since a missed churner is more costly than a false alarm) and experimented with lowering the classification threshold from 0.5 to 0.3 to quantify the precision/recall tradeoff

## Model Performance

| Metric | Threshold 0.5 (default) | Threshold 0.3 |
|---|---|---|
| Accuracy | 81.20% | 79.45% |
| Precision | 55.94% | 47.91% |
| Recall | 20.36% | 52.42% |
| F1 | 29.85% | 50.06% |

## Key Insights

- **A 99.9% accuracy result was a red flag, not a success** — it led to discovering a leaked feature (`Complain`) that was effectively encoding the target itself. Verifying suspiciously good results turned out to be as important as building the model in the first place.
- **After removing the leak, accuracy dropped to a much more realistic ~81%** — this number is trustworthy, where the earlier one was not.
- **Accuracy alone was misleading due to class imbalance.** With churners making up only ~20% of the dataset, the default model achieved 81% accuracy while catching just 20% of actual churners (Recall) — a model that would be close to useless for a bank trying to intervene before customers leave.
- **For a churn use case, Recall matters more than Precision**: missing an actual churner means silently losing real revenue, while a false alarm on a loyal customer costs little (at most an unnecessary retention offer).
- **Lowering the classification threshold from 0.5 to 0.3 roughly doubled Recall (20% → 52%)**, at the cost of lower Precision (56% → 48%) — a clear, quantified illustration of the precision-recall tradeoff. Choosing the "right" threshold in practice would require actual business input on the cost of each type of error, not just a mathematical answer.

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

- Try class-weighting or resampling techniques (e.g. SMOTE) to address the imbalance directly, rather than only adjusting the threshold
- Test a non-linear model (Random Forest, XGBoost) to see if it captures churn patterns more effectively
- Systematically test a range of thresholds and plot a precision-recall curve rather than comparing just two fixed values

## Notes

This project is part of a structured self-study path moving from data analysis through machine learning fundamentals into deep learning.