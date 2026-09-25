    # Credit Card Customer Segmentation — K-Means Clustering

Segmenting credit card customers into behaviorally distinct groups using unsupervised learning, with no labels or target variable — letting the natural structure in spending, borrowing, and repayment behavior emerge on its own.

> 🎓 This is my first unsupervised learning project, and the first where there's no "correct answer" to evaluate against — success here is judged by the internal structure of the clusters themselves (tightness and separation), not by comparison to a known label.

## Overview

This project applies K-Means clustering to a credit card customer dataset to identify distinct behavioral segments based on spending, cash advance usage, credit limit, and repayment patterns. The number of clusters was chosen using two independent methods — the Elbow Method and Silhouette Score — which agreed on the same result, giving more confidence in the final choice than either method alone would have.

## Tools Used

- Python
- Pandas, NumPy
- Scikit-learn (KMeans, StandardScaler, silhouette_score)
- Matplotlib

## Dataset

**Source:** Credit Card Dataset for Clustering (Kaggle)

Customer-level credit card usage data including balance, purchase behavior (one-off and installment), cash advance activity, credit limit, payments, minimum payments, and percentage of full-payment months.

## Key Steps Performed

- Loaded and inspected the dataset; dropped the customer ID column
- Handled missing values deliberately rather than defaulting to one method for every column:
  - Investigated whether `MINIMUM_PAYMENTS` nulls corresponded to customers with zero balance (which would justify filling with 0) — found this wasn't the case, so used median imputation instead, which better reflects a genuine data-collection gap rather than a meaningful zero
  - Filled `CREDIT_LIMIT`'s small number of missing values with the median
- Selected 7 features spanning four distinct behavioral dimensions — spending (`BALANCE`, `PURCHASES`), borrowing (`CASH_ADVANCE`), credit capacity (`CREDIT_LIMIT`), and repayment (`PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`) — deliberately avoiding redundant columns (e.g. not including `PURCHASES` alongside its own sub-components)
- Scaled all features with `StandardScaler`, required since K-Means is a distance-based algorithm
- Used the Elbow Method (inertia across K=1–10) to identify candidate values of K, but found the curve ambiguous, with plausible bends at both K=3 and K=5
- Calculated Silhouette Scores for the candidate K values, which clearly favored K=3 (0.44) over K=4, K=5, and K=6 — resolving the ambiguity the elbow method left open
- Fit the final K-Means model with K=3 and profiled each cluster's average values across all 7 features to interpret what distinguishes them

## Cluster Profiles

| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| Balance | $4,095 | $813 | $4,868 |
| Purchases | $10,318 | $758 | $1,187 |
| Cash Advance | $4,139 | $437 | $3,218 |
| Cash Advance Frequency | 0.18 | 0.10 | **0.31** |
| Credit Limit | $11,874 | $3,470 | $8,581 |
| Payments | $16,162 | $1,133 | $3,063 |
| % Full Payment | **0.38** | 0.18 | **0.02** |

**Cluster 0 — High-Value Active Spenders**: highest purchases, credit limit, and payments by a wide margin, combined with the healthiest full-payment rate. The most valuable, lowest-risk segment.

**Cluster 1 — Low-Activity / Dormant Customers**: lowest across nearly every metric — low balance, low purchases, low cash advance use. Customers who barely engage with their card.

**Cluster 2 — Cash-Advance Reliant, High-Risk**: a high balance similar to Cluster 0, but very low purchases and the highest cash advance frequency of any group — combined with a full-payment rate of just ~2%. This segment borrows heavily via cash advances and rarely clears its balance, a clear credit-risk signal.

## Key Insights

- **K-Means uncovered three behaviorally distinct, business-relevant customer segments** purely from spending and repayment patterns, with no labels provided at any point.
- **The elbow method alone was ambiguous** for this richer, 7-feature dataset (unlike a simpler 2-feature teaching dataset), showing plausible bends at both K=3 and K=5 — silhouette scores provided the deciding, more confident signal.
- **Cluster 2 is the most actionable finding**: customers with a high balance, heavy reliance on cash advances, and a near-zero full-payment rate represent a distinct high-risk segment that would likely warrant closer credit monitoring in a real business setting.
- **Cluster 0 and Cluster 2 have similar balances but completely different underlying behavior** — one driven by heavy purchasing with healthy repayment, the other driven by cash advances with poor repayment. This shows why balance alone would be a poor segmentation variable; the *combination* of features is what reveals meaningfully different customer types.
- **Cluster 1 represents low-engagement customers** across every metric, a segment more suited to re-engagement offers than credit risk concern.

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

- Use PCA to visualize all 7 features' cluster separation in 2D, rather than relying on just two features at a time for plotting
- Test whether K=5 (the other plausible elbow point) reveals more granular, still-meaningful sub-segments worth acting on separately
- Bring in additional behavioral features (e.g. tenure, purchase frequency patterns) to see if they refine the risk segment further

## Notes

This project is part of a structured self-study path moving from data analysis through machine learning fundamentals into deep learning. It marks the completion of the core supervised and unsupervised algorithms covered in the machine learning phase of that path.