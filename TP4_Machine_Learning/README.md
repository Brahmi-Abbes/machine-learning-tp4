# Practical Session 4: Overfitting, Data Leakage, and Model Generalization

**Course:** Machine Learning (Academic Year 2026–2027)  
**Level:** First-year Master's Degree  
**Submission Folder:** `TP4_Machine_Learning/`  

---

## 1. Assignment Objectives

The objective of Practical Session 4 is to investigate the real-world challenges of **data leakage**, **overfitting**, and **model generalization** using realistic survey data:
1. **Part A (Data Inspection & Potential Leakage):** Inspect survey distributions, detect missing values and duplicate rows, convert timestamps, evaluate high-cardinality cohort features, and detect potential target leakage (`performance_risk_level`).
2. **Part B (Impact of Data Leakage):** Quantify the empirical harm of:
   - Random vs. chronological splitting on time-dependent data.
   - Duplicate submissions leaking across training and test splits.
   - Preprocessing leakage (fitting `StandardScaler` globally vs. inside a pipeline on training data only).
3. **Part C (Understanding & Diagnosing Overfitting):** Diagnose overfitting caused by model complexity (decision tree maximum depth) and feature complexity (polynomial expansions of `effort_index`).
4. **Part D (Improving Generalization & Model Selection):** Apply leakage-safe validation (`TimeSeriesSplit`) on training data to:
   - Handle high-cardinality `cohort_code` (rare-category grouping vs. target encoding vs. feature removal).
   - Perform feature selection regarding temporal feature `submission_day`.
   - Apply L1/L2 regularization (Lasso and Ridge) to penalize higher-order polynomials.
   - Tune Decision Tree depth and implement chronological early stopping for Gradient Boosting using `staged_predict`.
   - Evaluate final models on the untouched chronological test set.
5. **Final Discussion:** Answer the professor's 6 conceptual questions grounded directly in actual computed results.

---

## 2. Dataset Description

- **File:** `hybrid_student_performance_1200.csv`
- **Observations:** 1,200 survey responses collected between 15 March and 14 April 2026.
- **Target Variable:** `final_cgpa`, converted from `cgpa_category` using the midpoint of each band:
  - `5.0–6.9` $\to$ 5.95
  - `7.0–8.4` $\to$ 7.70
  - `8.5–9.4` $\to$ 8.95
  - `9.5–10.0` $\to$ 9.75
- **Engineered Features:**
  - `study_hours`: Midpoint of `study_hours_daily` (0.5, 1.5, 2.5 hours).
  - `sleep_hours`: Midpoint (4.5, 6.5, 8.5 hours).
  - `attendance_rate`: Midpoint of `attendance_percentage` (40.0%, 57.5%, 70.5%, 80.5%, 92.5%).
  - `study_consistency`: Ordinal 0–2 (Rarely, Sometimes, Mostly consistent).
  - `tasks_on_time`, `assignments_on_time`: Ordinal 0–3 (Rarely, Sometimes, Often, Always).
  - `revision_frequency`: Ordinal 0–3 (Never, Rarely, Few times a week, Daily).
  - `career_goal_clarity`: Ordinal 0–2 (Not clear, Somewhat clear, Very clear).
  - `daily_productivity`, `stress_level`, `age`: Raw numerical scores.
  - `cohort_code`: Interaction string `program_stream + " | " + year_class` (42 unique categories, 12 with $<10$ observations).
  - `effort_index`: Unweighted row-wise mean of 6 normalized effort components, ignoring missing answers within each row.
  - `submission_day`: Continuous elapsed days from the earliest survey submission.
- **Excluded Features:**
  - `student_id`: Arbitrary identifier.
  - `performance_risk_level`: Administrative label derived directly from CGPA; causes severe target leakage.

---

## 3. Environment & Requirements

The project uses Python 3.10+ (tested on Python 3.14) and standard scientific libraries:
- `numpy>=1.26.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.4.0`
- `matplotlib>=3.7.0`

### Installation
From within the repository, install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 4. How to Execute the Script

Run the complete pipeline from within the `TP4_Machine_Learning/` folder:
```bash
cd TP4_Machine_Learning
python tp4.py
```
*(Alternatively, you can run it from the repository root: `python TP4_Machine_Learning/tp4.py`)*

The script will automatically:
1. Load `hybrid_student_performance_1200.csv` via relative path.
2. Run Parts A, B, C, D, and print comprehensive step-by-step progress to the terminal.
3. Save all formatted tables as `.csv` in `results/`.
4. Save all generated figures as `.png` in `results/plots/`.
5. Output the complete final discussion to `results/final_discussion.txt`.

---

## 5. Summary of Experimental Results

### Part B: Data Leakage Experiments
| Evaluation Strategy | Model | Training RMSE | Testing RMSE |
|---|---|:---:|:---:|
| **B.1 Random Split (80/20)** | Decision Tree (unpruned) | 0.0000 | 1.5785 |
| **B.1 Chronological Split (80/20)** | Decision Tree (unpruned) | 0.0000 | 1.9750 |
| **B.1 Dummy Baseline (Chronological)** | DummyRegressor (mean) | 1.2685 | 1.2362 |
| **B.2 Duplicate Leakage (Split Before Dedup)** | Decision Tree (unpruned) | 0.0000 | 1.8257 |
| **B.2 Clean (Dedup Before Split)** | Decision Tree (unpruned) | 0.0000 | 1.7127 |
| **B.3 Preprocessing Leakage (Global Scaler)** | KNN ($k=10$) | 1.0691 | 1.1569 |
| **B.3 Clean Preprocessing (Pipeline Scaler)** | KNN ($k=10$) | 1.0658 | 1.1469 |

*Key Findings:*
- In **B.1**, random splitting leaks future temporal distribution patterns, yielding an artificially optimistic test RMSE (1.5785 vs 1.9750). Both unpruned trees overfit drastically, scoring worse than the dummy baseline (1.2362).
- In **B.2**, 52 duplicated students spanned across training and testing splits, artificially distorting out-of-sample evaluation.
- In **B.3**, global scaling leaks test distribution statistics into the training phase; pipelines prevent this data snooping.

---

### Part C: Overfitting Diagnosis
- **C.1 Model Complexity (Decision Tree):**
  - `max_depth = 1`: Train RMSE = 1.1794, Test RMSE = 1.1594, Gap = -0.0200
  - `max_depth = 2`: Train RMSE = 1.1394, Test RMSE = 1.1268, Gap = -0.0126
  - `max_depth = 3`: Train RMSE = 1.1021, **Test RMSE = 1.0677** (optimal depth)
  - `max_depth = 5`: Train RMSE = 1.0087, Test RMSE = 1.3352, Gap = +0.3265 (overfitting starts)
  - `max_depth = 10`: Train RMSE = 0.5388, Test RMSE = 1.9358, Gap = +1.3970
  - `max_depth = 20 / None`: Train RMSE = 0.0000, Test RMSE = 1.9750, Gap = +1.9750
- **C.2 Feature Complexity (Polynomial Degrees of effort_index):**
  - Validation error degrades monotonically as degree increases (CV RMSE: Degree 1 = 1.1774 $\to$ Degree 15 = 1.2941). The underlying relationship is linear; higher degrees fit noise at the bounds.

---

### Part D: Final Model Comparison (Untouched Chronological Test Set)
Evaluated on the final feature matrix (Part B features excluding `submission_day` and `cohort_code`):

| Model | Training RMSE | Testing RMSE | Testing $R^2$ |
|---|:---:|:---:|:---:|
| **Baseline (training mean)** | 1.2685 | 1.2362 | -0.0029 |
| **Linear Regression** | 1.0897 | 1.0576 | 0.2659 |
| **Ridge Regression ($\alpha=100.0$)** | 1.0908 | 1.0591 | 0.2639 |
| **Decision Tree ($\text{max\_depth}=3$)** | 1.1021 | 1.0677 | 0.2519 |
| **Gradient Boosting ($n=48$ via early stopping)** | **1.0454** | **1.0573** | **0.2664** |

---

## 6. Generated Output Files

### Generated Plots (`results/plots/`)
- `part_a_rolling_cgpa.png`: 7-day rolling mean of student CGPA over the survey period.
- `part_c1_tree_depth_overfitting.png`: Training vs. testing error against decision tree maximum depth.
- `part_c2_polynomial_degree_cv.png`: 5-fold TimeSeriesSplit CV error against polynomial degree of `effort_index`.
- `part_d3_regularization.png`: Validation curves for Ridge and Lasso across regularization parameter $\alpha$.
- `part_d4_final_models_comparison.png`: Final model comparison bar chart (Training/Testing RMSE and Testing $R^2$).

### Generated Result Tables (`results/`)
- `part_b_summary.csv`: Summary of leakage experiments B.1, B.2, B.3.
- `part_c1_complexity.csv`: Full depth table for tree complexity.
- `part_c2_feature_complexity.csv`: Cross-validation scores for polynomial degrees.
- `part_d1_cohort_encoding.csv`: Comparison of high-cardinality categorical treatments.
- `part_d2_feature_selection.csv`: Impact of `submission_day` across multiple algorithms.
- `part_d3_regularization_coefficients.csv`: Degree-5 polynomial coefficients for LR, Ridge, and Lasso.
- `final_model_comparison.csv`: Final evaluation table on the untouched test set.
- `final_discussion.txt`: Complete answers to the professor's 6 discussion questions.

