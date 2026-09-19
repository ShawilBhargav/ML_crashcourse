# Medical Insurance Cost Prediction — Linear Regression

Predicting individual medical insurance charges using demographic and lifestyle features, with a focus on understanding *where* and *why* a simple linear model succeeds or fails.

> 🎓 This is my second ML project, built while working through machine learning fundamentals after completing an EDA project on Netflix's catalog. This one focuses on the full supervised learning pipeline — from encoding categorical features through training, evaluation, and diagnosing model errors visually.

## Overview

This project builds a Linear Regression model to predict medical insurance charges based on age, BMI, number of children, sex, smoking status, and region. Beyond just training a model and reporting accuracy, the project digs into *why* the model performs the way it does — using residual analysis to uncover a hidden pattern in the errors, and confirming the cause visually.

## Tools Used

- Python
- Pandas, NumPy
- Scikit-learn (LinearRegression, train_test_split, metrics)
- Matplotlib, Seaborn

## Dataset

**Source:** [Medical Cost Personal Datasets — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

1,338 records with features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and the target `charges` (insurance cost in USD).

## Key Steps Performed

- Loaded and inspected the dataset — confirmed no missing values, identified categorical columns (`sex`, `smoker`, `region`)
- Encoded categorical features using one-hot encoding (`pd.get_dummies`, `drop_first=True`) to avoid the dummy variable trap
- Split data into training (80%) and test (20%) sets
- Trained a Linear Regression model using scikit-learn
- Evaluated performance using MAE, RMSE, and R²
- Visualized predicted vs. actual charges to assess overall model fit
- Built a residual plot (errors vs. predicted values) to check for non-random error patterns
- Identified structured banding in the residuals and confirmed, by coloring the plot with `hue=smoker`, that the pattern was driven by smoking status

## Model Performance

| Metric | Value |
|---|---|
| MAE | ~$4,181 |
| MSE | ~$33.6 Million |
| R² | 0.78 |

## Key Insights

- **Smoker status has a major, non-linear effect on insurance charges.** A plain linear model systematically misjudges predictions differently for smokers vs. non-smokers, since it's forced to fit one straight-line relationship across a population with two very different cost behaviors.
- **The model explains ~78% of the variance in charges (R² = 0.78)** using just six features — smoker status is very likely the single largest driver of this predictive power.
- **MSE (33.6 Million Dollar) is noticeably higher than MAE (4,181 Dollar)**, signaling that a subset of predictions carry much larger errors than the rest — later confirmed to be concentrated among smokers.
- **Residual analysis revealed structured banding rather than random scatter** — a clear sign the model is missing something systematic, not just noisy. Random, unstructured residuals are what you'd expect from a well-fit model; this wasn't that.
- **Coloring the residual plot by smoker status confirmed the hypothesis** — the bands separated cleanly along smoker vs. non-smoker lines, turning a visual hunch into an evidence-backed finding.
- **High-BMI smokers are the hardest cases to predict accurately**, with the model tending to underpredict extreme high-charge cases.

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Download the dataset from Kaggle and place `insurance.csv` in the project folder
4. Open `prac.py` in Jupyter Notebook or Google Colab
5. Run all cells in order

## What I'd Improve Next

- Try a non-linear model (e.g. Random Forest) to see if it resolves the residual banding
- Add an interaction term between `smoker` and `bmi`, since their combined effect appears to drive the largest errors
- Investigate the single major outlier with a very high positive residual

## Notes

This project builds directly on fundamentals covered in an earlier EDA project and is part of a structured self-study path moving from data analysis into machine learning and deep learning.