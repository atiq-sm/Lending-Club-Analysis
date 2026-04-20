# Lending Club Analysis

Machine learning analysis of Lending Club loans focused on:
- **Supervised grade prediction** (binary classification)
- **Feature explainability** (LIME + SHAP + generated text summaries)
- **Unsupervised clustering** to study whether natural groups align with loan grades

This repository is notebook-driven and contains two end-to-end analyses on the Lending Club 2014–2016 dataset.

---

## Repository Structure

```text
Lending Club Classification.ipynb   # Supervised modeling + explainability
Lending Club Clustering.ipynb       # Dimensionality reduction + clustering analysis
README.md
```

---

## Dataset

Both notebooks load the same file from the project root:

- `lc_14to16.csv.zip`

Expected load command used in notebooks:

```python
pd.read_csv("lc_14to16.csv.zip", compression="zip")
```

> Place the dataset zip in the repository root before running notebooks.

---

## Project 1: Classification (`Lending Club Classification.ipynb`)

### Goal
Predict a binary form of loan grade and compare multiple ML classifiers.

### Pipeline Summary
1. Load Lending Club data and remove many irrelevant/high-missing columns.
2. Consolidate and encode categorical variables (e.g., employment length, home ownership, grade).
3. Build a binary modeling dataframe with selected numeric + encoded features.
4. Handle missing values (median imputation) and standardize numeric features.
5. Detect outliers using multiple methods (Isolation Forest, Elliptic Envelope; LOF code present) and remove consensus outliers.
6. Remove extreme `annual_inc` and `dti` tails after scaling-based filtering.
7. Train/test split and model comparison.
8. Save model packages (`*.pkl`) and run explainability workflow.

### Models Trained
- K-Nearest Neighbors
- SGD Classifier
- Random Forest
- Gradient Boosting Trees
- XGBoost

### Reported Test Performance
| Model | Accuracy |
|---|---:|
| KNN | 0.7500 |
| SGD | 0.7600 |
| Random Forest | 0.7756 |
| Gradient Boosting Trees | 0.7987 |
| XGBoost | 0.7952 |

Best reported accuracy in the notebook: **Gradient Boosting Trees (~0.80)**.

### Explainability
The notebook includes:
- **LIME tabular explanations**
- **SHAP-based feature contributions**
- A **text explanation pipeline** (Transformers/Hugging Face) to summarize model behavior for selected instances.

---

## Project 2: Clustering (`Lending Club Clustering.ipynb`)

### Goal
Discover latent borrower/loan groups and test whether clusters align with labeled grades.

### Pipeline Summary
1. Load and time-split data by `issue_d` (before/after 2015) for analysis scope.
2. Remove unrelated columns and preprocess/encode features.
3. Filter columns by missing-value threshold and impute remaining nulls.
4. Standardize numeric data.
5. Outlier handling with LOF, Isolation Forest, and Elliptic Envelope.
6. Dimensionality reduction with:
   - PCA
   - UMAP (2D and 3D embeddings)
7. Clustering experiments with:
   - K-Means on UMAP 2D
   - K-Means on UMAP 3D
   - K-Means on PCA space
   - HDBSCAN
8. Statistical post-analysis (ANOVA/F-tests on key financial features across clusters).

### Main Finding
Across clustering variants, the notebook concludes:
- weak/no strong alignment between discovered clusters and labeled grades
- limited significant variance in key features across clusters

Implication: supervised approaches are more informative than unsupervised cluster structure for grade prediction in this feature space.

---

## Environment & Dependencies

The notebooks use Python 3 and common DS/ML libraries, including:
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- xgboost
- shap, lime
- umap-learn, hdbscan, scipy, plotly
- transformers, huggingface_hub
- torch, tensorflow

Since no pinned environment file is included, install dependencies manually (or via your own environment manager) before execution.

---

## How to Run

1. Create/activate a Python 3 environment.
2. Install required libraries used in the notebooks.
3. Place `lc_14to16.csv.zip` in the repository root.
4. Launch Jupyter and run notebooks in order:
   1. `Lending Club Classification.ipynb`
   2. `Lending Club Clustering.ipynb`

---

## Notes

- This repository is analysis-first and notebook-centric.
- Trained model artifacts (`*.pkl`) are generated during notebook execution and may not be committed.
- Some explainability/text-generation sections rely on external model access (Hugging Face authentication may be required).
