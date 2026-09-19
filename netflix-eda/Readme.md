# Netflix Movies & TV Shows — Exploratory Data Analysis

Analyzing Netflix's content catalog to uncover trends in content type, ratings, genres, and global production patterns.

> 🎓 This is my first end-to-end EDA project, built while learning data analysis from scratch as part of my journey into ML/AI/Deep Learning. I'm sharing it as-is — a genuine first step, not a polished final product — and plan to revisit and expand it as I learn more.

## Overview

This project explores the Netflix Movies and TV Shows dataset (sourced from Kaggle) to understand how Netflix's content catalog has evolved — including the balance between movies and TV shows, dominant content ratings, top-producing countries, and genre trends. The analysis covers data cleaning, exploratory visualization, and derived insights using Python.

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Dataset

**Source:** [Netflix Movies and TV Shows — Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)

~8,800 titles with attributes including type, director, cast, country, date added, release year, rating, duration, and genre (`listed_in`).

## Key Steps Performed

- Explored the raw dataset structure (`.info()`, `.describe()`, `.shape`) to understand data types and scale
- Cleaned missing values in `director`, `cast`, and `country` using informed fill strategies (`"Unknown"` / `"Not Specified"`) rather than dropping rows, to preserve data
- Dropped rows with missing `date_added` and `rating`, where the missing count was small enough that dropping had minimal impact
- Split the mixed-unit `duration` column into two purpose-built columns — `duration_min` (for Movies) and `duration_season` (for TV Shows) — using conditional assignment based on `type`
- Checked for and confirmed no duplicate rows or duplicate titles
- Performed univariate analysis on `type`, `rating`, and `release_year`
- Performed bivariate analysis: content growth over time by `type`, top content-producing countries, top genres, and rating distribution across `type`
- Handled multi-value columns (`country`, `listed_in`) by splitting and exploding comma-separated values before counting, to avoid miscounting combined entries as single categories

## Key Insights

- **Movies dominate the catalog**, making up the majority of titles compared to TV Shows — and their share has grown faster than TV Shows over the past decade.
- **TV-MA is the most common content rating**, both overall and within Movies and TV Shows individually — indicating a catalog skewed toward mature audiences.
- **The United States leads content production** by a clear margin, though the catalog spans many countries globally.
- **Movies and TV Shows use distinct rating systems** — MPAA-style ratings (G, PG, PG-13, R, NC-17) apply almost exclusively to Movies, while TV-style ratings (TV-Y, TV-14, TV-MA) apply almost exclusively to TV Shows, with near-zero overlap between the two systems.
- **Most content in the catalog is recent**, with release years right-skewed toward the last decade — reflecting Netflix's rapid content expansion rather than a deep back-catalog of older titles.
- International Movies, Dramas, and Comedies are among the most common genres in the catalog.

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn
   ```
3. Open `netflix_eda.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells in order

## What I'd Improve Next

As my first project, there's plenty of room to build on this:
- Genre-based content recommendation exploration
- Predicting content rating from other features (classification)
- Deeper geographic analysis (country-level genre/rating preferences)
- Interactive dashboard version using Plotly or Streamlit

## Notes

This project was built as part of a structured self-study plan covering data analysis fundamentals before moving into machine learning and deep learning. Feedback and suggestions are welcome!