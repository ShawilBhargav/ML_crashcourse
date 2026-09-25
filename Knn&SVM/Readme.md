# Wine Quality Prediction — KNN vs. SVM (Multi-Class Classification)

Comparing KNN and SVM (linear and RBF kernels) on a multi-class wine quality prediction task, with a specific focus on how overall accuracy can hide poor performance on rare classes — and why the choice of averaging method for precision/recall/F1 matters as much as the model choice itself.

> 🎓 This project intentionally tackles a genuine multi-class problem rather than binarizing the target, in order to practice macro vs. weighted metric averaging — a distinction that turned out to be the most important finding in the whole project.

## Overview

This project predicts wine quality (a multi-class target ranging across several score levels) from physicochemical properties like acidity, sugar, sulfur dioxide, density, pH, and alcohol content. Three models — KNN, SVM with an RBF kernel, and SVM with a linear kernel — were trained and compared, with every model evaluated using **both macro and weighted averaging** for precision, recall, and F1, to reveal whether strong-looking accuracy was actually masking poor performance on rare quality classes.

## Tools Used

- Python
- Pandas, NumPy
- Scikit-learn (KNeighborsClassifier, SVC, StandardScaler, train_test_split, metrics)
- Matplotlib

## Dataset

**Source:** Wine Quality dataset (Kaggle)

1,143 records with 11 physicochemical features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol) and a multi-class `quality` target (integer score), plus a non-predictive `Id` column.

## Key Steps Performed

- Loaded and inspected the dataset; dropped the `Id` column (pure identifier, no predictive value)
- Confirmed `quality` is a genuine multi-class target with an imbalanced distribution — most wines cluster around the middle quality scores, with very few at the extremes
- Split data into training (80%) and test (20%) sets
- Scaled features with `StandardScaler` (fit on training data only), required for both KNN and SVM since both are distance/margin-based algorithms sensitive to feature scale
- Trained a KNN classifier and evaluated using both macro and weighted averaging for precision, recall, and F1
- Trained SVM with an RBF kernel and a separate SVM with a linear kernel, evaluating both the same way
- Compared all three models directly, with particular attention to the gap between macro and weighted scores for each

## Model Performance

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---|---|---|---|---|---|
| KNN | 52.84% | 27.41% | 25.40% | 26.27% | 54.38% | 52.84% | 53.52% |
| SVM (RBF) | **63.76%** | **36.65%** | **33.91%** | **34.31%** | 61.15% | 63.76% | 61.77% |
| SVM (Linear) | 62.01% | 24.92% | 29.15% | 26.86% | 52.99% | 62.01% | 57.12% |

## Key Insights

- **SVM with an RBF kernel was the strongest model overall**, winning on accuracy and, critically, on macro-averaged metrics too — meaning it wasn't just favoring the common classes, it genuinely handled the full range of quality scores better than the alternatives.
- **The gap between macro and weighted scores was large and consistent across every model** (roughly 25-30 percentage points), revealing that overall accuracy and weighted metrics substantially overstate real-world usefulness. All three models perform reasonably on common, mid-range quality wines but struggle badly — sometimes completely failing — on rare, extreme-quality wines.
- **SVM (Linear) is a clear example of this failure mode in its rawest form**: despite a respectable 62% accuracy, its confusion matrix shows it never once correctly predicted either of the two rarest quality classes — the last two columns of its confusion matrix are entirely zero. It achieved a "good-looking" accuracy by essentially giving up on rare classes entirely.
- **Switching the `average` parameter from `'weighted'` to `'macro'` changes the story completely**, even though nothing about the model or data changes — it's purely a difference in how per-class scores get combined into one number. Weighted averaging lets common classes dominate the score (masking rare-class failure); macro averaging treats every class equally, honestly exposing it. This is a reminder that metric choice is not a neutral technical detail — it directly determines what conclusion a report reaches.
- **For a real-world quality-detection system**, the extreme classes (very poor or exceptional wines) are arguably the most business-relevant cases to catch correctly — making macro-averaged metrics the more honest, relevant measure here, not accuracy or weighted scores.
- **RBF's advantage over the linear kernel mirrors an earlier finding on a different dataset** (Titanic survival prediction), reinforcing that wine quality, like many real-world targets, isn't cleanly separable by a straight decision boundary.

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib
   ```
3. Download the dataset from Kaggle and place the CSV in the project folder
4. Open the notebook in Jupyter Notebook or Google Colab
5. Run all cells in order

## What I'd Improve Next

- Address the class imbalance directly (e.g. class weighting, oversampling rare classes with SMOTE) rather than only diagnosing it after the fact
- Try a tree-based ensemble (Random Forest) to see if it handles the rare classes better than KNN/SVM
- Collapse adjacent rare quality scores into broader bins to see if that produces a more practically useful classifier without discarding the extremes entirely

## Notes

This project is part of a structured self-study path moving from data analysis through machine learning fundamentals into deep learning.