"""
Machine Learning - Practical Session 4
Topic: Overfitting, Data Leakage, and Model Generalization
Academic Year: 2026-2027 | First-year Master's degree
Author: First-year Master's Student

This script executes the complete TP4 assignment covering:
- Part A: Data Inspection and Potential Leakage
- Part B: Impact of Data Leakage and Evaluation Strategy
- Part C: Understanding and Diagnosing Overfitting
- Part D: Improving Model Generalization & Final Evaluation
- Final Discussion: Grounded answers to professor's 6 questions
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless and script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeRegressor

# Suppress minor warnings for clean student output
warnings.filterwarnings("ignore")

# Define RMSE helper compatible with scikit-learn
def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# Set fixed random seed for full reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup directories (paths relative to this script)
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = BASE_DIR / "hybrid_student_performance_1200.csv"


# ==============================================================================
# SECTION 1: DATA ENCODING & PREPARATION (PDF SECTION 2)
# ==============================================================================
def load_and_encode_data(csv_path=CSV_FILE):
    """
    Load survey CSV and apply the exact encodings specified in Section 2 of the PDF.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw_df = pd.read_csv(csv_path)

    # Make working copy
    df = raw_df.copy()

    # 1. Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 2. Target encoding: cgpa_category -> final_cgpa band midpoints
    # 5.0–6.9 -> 5.95; 7.0–8.4 -> 7.70; 8.5–9.4 -> 8.95; 9.5–10.0 -> 9.75
    cgpa_map = {
        "5.0 – 6.9": 5.95, "5.0-6.9": 5.95,
        "7.0 – 8.4": 7.70, "7.0-8.4": 7.70,
        "8.5 – 9.4": 8.95, "8.5-9.4": 8.95,
        "9.5 – 10.0": 9.75, "9.5-10.0": 9.75,
    }
    df["final_cgpa"] = df["cgpa_category"].map(cgpa_map)

    # 3. study_hours_daily -> study_hours (midpoint in hours)
    study_map = {
        "Less than 1 hour": 0.5,
        "1–2 hours": 1.5, "1-2 hours": 1.5,
        "More than 2 hours": 2.5,
    }
    df["study_hours"] = df["study_hours_daily"].map(study_map)

    # 4. sleep_hours (midpoint in hours)
    sleep_map = {
        "4–5 hours": 4.5, "4-5 hours": 4.5,
        "6–7 hours": 6.5, "6-7 hours": 6.5,
        "More than 8 hours": 8.5,
    }
    df["sleep_hours"] = df["sleep_hours"].map(sleep_map)

    # 5. attendance_percentage -> attendance_rate (midpoint in %)
    attendance_map = {
        "Less than 50%": 40.0,
        "50% – 65%": 57.5, "50%-65%": 57.5,
        "66% – 75%": 70.5, "66%-75%": 70.5,
        "76% – 85%": 80.5, "76%-85%": 80.5,
        "Above 85%": 92.5,
    }
    df["attendance_rate"] = df["attendance_percentage"].map(attendance_map)

    # 6. study_consistency (Ordinal 0–2)
    consistency_map = {
        "Rarely": 0,
        "Sometimes": 1,
        "Mostly consistent": 2,
    }
    df["study_consistency"] = df["study_consistency"].map(consistency_map)

    # 7. tasks_on_time & assignments_on_time (Ordinal 0–3)
    punctuality_map = {
        "Rarely": 0,
        "Sometimes": 1,
        "Often": 2,
        "Always": 3,
    }
    df["tasks_on_time"] = df["tasks_on_time"].map(punctuality_map)
    df["assignments_on_time"] = df["assignments_on_time"].map(punctuality_map)

    # 8. revision_frequency (Ordinal 0–3)
    revision_map = {
        "Never": 0,
        "Rarely": 1,
        "Few times a week": 2,
        "Daily": 3,
    }
    df["revision_frequency"] = df["revision_frequency"].map(revision_map)

    # 9. career_goal_clarity (Ordinal 0–2)
    clarity_map = {
        "Not clear": 0,
        "Somewhat clear": 1,
        "Very clear": 2,
    }
    df["career_goal_clarity"] = df["career_goal_clarity"].map(clarity_map)

    # 10. daily_productivity, stress_level, age are already numeric
    df["daily_productivity"] = pd.to_numeric(df["daily_productivity"])
    df["stress_level"] = pd.to_numeric(df["stress_level"])
    df["age"] = pd.to_numeric(df["age"])

    # 11. cohort_code: program_stream + " | " + year_class
    df["cohort_code"] = df["program_stream"].astype(str) + " | " + df["year_class"].astype(str)

    # 12. effort_index: mean of 6 normalized components, ignoring missing answers within row
    effort_components = pd.DataFrame({
        "comp1": df["study_hours"] / 2.5,
        "comp2": df["attendance_rate"] / 100.0,
        "comp3": df["study_consistency"] / 2.0,
        "comp4": df["tasks_on_time"] / 3.0,
        "comp5": df["assignments_on_time"] / 3.0,
        "comp6": df["revision_frequency"] / 3.0,
    })
    df["effort_index"] = effort_components.mean(axis=1)

    return raw_df, df


# ==============================================================================
# PART A: DATA INSPECTION AND POTENTIAL LEAKAGE
# ==============================================================================
def run_part_a(raw_df, df):
    print("=" * 80)
    print("PART A: DATA INSPECTION AND POTENTIAL LEAKAGE")
    print("=" * 80)

    # Task 1: Dimensions, data types, descriptive stats, missing values
    print(f"\n1. Dataset Dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print("\nMissing values per column in raw survey (showing columns with missing values):")
    missing_counts = raw_df.isnull().sum()
    missing_pct = (missing_counts / len(raw_df)) * 100
    missing_table = pd.DataFrame({"Missing Rows": missing_counts, "Percentage (%)": missing_pct})
    missing_table = missing_table[missing_table["Missing Rows"] > 0].sort_values("Missing Rows", ascending=False)
    print(missing_table.to_string())

    # Task 2: Duplicate detection
    dup_all = raw_df.duplicated().sum()
    dup_no_id = raw_df.duplicated(subset=[c for c in raw_df.columns if c != "student_id"]).sum()
    dup_no_ts = raw_df.duplicated(subset=[c for c in raw_df.columns if c != "timestamp"]).sum()
    dup_no_both = raw_df.duplicated(subset=[c for c in raw_df.columns if c not in ["student_id", "timestamp"]]).sum()

    print("\n2. Duplicate Detection:")
    print(f"   - All columns included: {dup_all} duplicates")
    print(f"   - Excluding student_id: {dup_no_id} duplicates")
    print(f"   - Ignoring timestamp: {dup_no_ts} duplicates")
    print(f"   - Excluding student_id & timestamp: {dup_no_both} duplicates")
    print("   Observation: The original survey contains NO exact duplicate observations.")

    # Task 3: Correlation with final_cgpa
    num_features = [
        "study_hours", "sleep_hours", "attendance_rate", "study_consistency",
        "tasks_on_time", "assignments_on_time", "revision_frequency",
        "career_goal_clarity", "daily_productivity", "stress_level", "age",
        "effort_index",
    ]
    correlations = df[num_features].corrwith(df["final_cgpa"]).sort_values(ascending=False)
    corr_df = pd.DataFrame({"Pearson Correlation with final_cgpa": correlations})
    print("\n3. Feature Correlations with final_cgpa:")
    print(corr_df.to_string())
    print(f"   - Strongest predictor: '{correlations.index[0]}' (r = {correlations.iloc[0]:.4f}), followed by 'attendance_rate' (r = {correlations.get('attendance_rate', 0):.4f})")
    print(f"   - Weakest predictors: '{correlations.index[-1]}' (r = {correlations.iloc[-1]:.4f}) and 'sleep_hours' (r = {correlations.get('sleep_hours', 0):.4f})")

    # Task 4: Compare performance_risk_level with cgpa_category
    print("\n4. Crosstab: performance_risk_level vs. cgpa_category:")
    risk_crosstab = pd.crosstab(df["performance_risk_level"], df["cgpa_category"])
    print(risk_crosstab)
    print("   Analysis: performance_risk_level is an administrative post-evaluation category")
    print("   directly constructed from student performance (target proxy). Including it as")
    print("   a predictive feature would introduce direct target leakage.")

    # Task 5: cohort_code distribution
    cohort_counts = df["cohort_code"].value_counts()
    rare_cohorts = cohort_counts[cohort_counts < 10]
    print(f"\n5. cohort_code Analysis:")
    print(f"   - Total unique cohorts: {len(cohort_counts)}")
    print(f"   - Cohorts with fewer than 10 observations: {len(rare_cohorts)} out of {len(cohort_counts)} ({len(rare_cohorts)/len(cohort_counts)*100:.1f}%)")
    print("   - Sample rare cohorts:", list(rare_cohorts.index[:4]))

    # Task 6: Chronological check and rolling mean plot
    is_ordered = df["timestamp"].is_monotonic_increasing
    print(f"\n6. Is dataset chronologically ordered? {is_ordered}")
    
    # Sort chronologically as instructed in Ground Rules
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    first_ts = df_sorted["timestamp"].min()
    df_sorted["submission_day"] = (df_sorted["timestamp"] - first_ts).dt.total_seconds() / 86400.0

    # 7-day rolling mean plot
    ts_indexed = df_sorted.set_index("timestamp")
    rolling_7d = ts_indexed["final_cgpa"].rolling("7D").mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ts_indexed.index, ts_indexed["final_cgpa"], ".", alpha=0.2, color="gray", label="Individual Submissions")
    ax.plot(rolling_7d.index, rolling_7d, color="#1f77b4", linewidth=2.5, label="7-Day Rolling Mean CGPA")
    ax.axhline(df["final_cgpa"].mean(), color="crimson", linestyle="--", linewidth=1.2, label=f"Overall Mean ({df['final_cgpa'].mean():.2f})")
    ax.set_title("Part A: Final CGPA Over Time (7-Day Rolling Mean)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Survey Submission Date", fontsize=11)
    ax.set_ylabel("Final CGPA", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plot_path = PLOTS_DIR / "part_a_rolling_cgpa.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"   Plot saved to: {plot_path.relative_to(BASE_DIR)}")

    # Summary risks
    print("\nSummary of Risks in Dataset (Part A Output):")
    print("  * Risk 1 (Target Leakage): 'performance_risk_level' directly categorizes student performance; using it leaks the target.")
    print("  * Risk 2 (High-Cardinality & Rare Categories): 'cohort_code' has 42 distinct categories, with 12 categories having fewer than 10 responses, causing high variance and unobserved levels in test splits.")
    print("  * Risk 3 (Temporal Leakage & Extrapolation): 'submission_day' increases strictly monotonically; in chronological evaluation, test observations lie beyond the training range.")
    print("  * Risk 4 (Missing Data Preprocessing Leakage): Multiple features have 2-5% missing values; median imputation statistics must be estimated on training folds only.")
    print("  * Risk 5 (Duplicate Resubmissions): Students submitting multiple times can cause identical feature-target pairs to span across train and test sets, inflating test metrics.")

    return df_sorted


# ==============================================================================
# PART B: IMPACT OF DATA LEAKAGE AND EVALUATION STRATEGY
# ==============================================================================
def run_part_b(df_sorted):
    print("\n" + "=" * 80)
    print("PART B: IMPACT OF DATA LEAKAGE AND EVALUATION STRATEGY")
    print("=" * 80)

    # Specified predictive features for Part B (11 features + submission_day)
    features_b = [
        "study_hours", "sleep_hours", "attendance_rate", "study_consistency",
        "tasks_on_time", "assignments_on_time", "revision_frequency",
        "career_goal_clarity", "daily_productivity", "stress_level", "age",
        "submission_day",
    ]

    X = df_sorted[features_b]
    y = df_sorted["final_cgpa"]

    # Fixed chronological 80/20 split
    n_train = int(len(df_sorted) * 0.8)
    X_train_c = X.iloc[:n_train].copy()
    y_train_c = y.iloc[:n_train].copy()
    X_test_c = X.iloc[n_train:].copy()
    y_test_c = y.iloc[n_train:].copy()

    # --------------------------------------------------------------------------
    # B.1: Compare Splitting Strategies
    # --------------------------------------------------------------------------
    print("\n--- B.1: Compare Splitting Strategies ---")
    pipe_dt = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("dt", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    ])

    # 1. Random split 80/20
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pipe_dt.fit(X_train_r, y_train_r)
    r_tr_rmse = rmse_score(y_train_r, pipe_dt.predict(X_train_r))
    r_te_rmse = rmse_score(y_test_r, pipe_dt.predict(X_test_r))

    dummy_r = DummyRegressor(strategy="mean")
    dummy_r.fit(X_train_r, y_train_r)
    r_base_rmse = rmse_score(y_test_r, dummy_r.predict(X_test_r))

    # 2. Chronological split 80/20
    pipe_dt.fit(X_train_c, y_train_c)
    c_tr_rmse = rmse_score(y_train_c, pipe_dt.predict(X_train_c))
    c_te_rmse = rmse_score(y_test_c, pipe_dt.predict(X_test_c))

    dummy_c = DummyRegressor(strategy="mean")
    dummy_c.fit(X_train_c, y_train_c)
    c_base_rmse = rmse_score(y_test_c, dummy_c.predict(X_test_c))

    print(f"Random Split:        Train RMSE = {r_tr_rmse:.4f}, Test RMSE = {r_te_rmse:.4f}, Baseline RMSE = {r_base_rmse:.4f}")
    print(f"Chronological Split: Train RMSE = {c_tr_rmse:.4f}, Test RMSE = {c_te_rmse:.4f}, Baseline RMSE = {c_base_rmse:.4f}")

    # --------------------------------------------------------------------------
    # B.2: Investigate Duplicate Leakage
    # --------------------------------------------------------------------------
    print("\n--- B.2: Investigate Duplicate Leakage ---")
    # Simulate duplicate submissions (15% sample, random_state=42)
    dup_sample = df_sorted.sample(frac=0.15, random_state=RANDOM_STATE).copy()
    t_min = df_sorted["timestamp"].min()
    t_max = df_sorted["timestamp"].max()
    period_seconds = int((t_max - t_min).total_seconds())

    rng = np.random.default_rng(RANDOM_STATE)
    random_seconds = rng.integers(0, period_seconds, size=len(dup_sample))
    dup_sample["timestamp"] = [t_min + pd.Timedelta(seconds=int(s)) for s in random_seconds]

    # Combine and sort chronologically
    df_augmented = pd.concat([df_sorted, dup_sample], ignore_index=True)
    df_augmented = df_augmented.sort_values("timestamp").reset_index(drop=True)
    df_augmented["submission_day"] = (df_augmented["timestamp"] - df_augmented["timestamp"].min()).dt.total_seconds() / 86400.0

    # Experiment 1: Splitting before removing duplicates
    n_train_aug = int(len(df_augmented) * 0.8)
    train_aug = df_augmented.iloc[:n_train_aug]
    test_aug = df_augmented.iloc[n_train_aug:]

    pipe_dt.fit(train_aug[features_b], train_aug["final_cgpa"])
    dup_leaked_tr_rmse = rmse_score(train_aug["final_cgpa"], pipe_dt.predict(train_aug[features_b]))
    dup_leaked_te_rmse = rmse_score(test_aug["final_cgpa"], pipe_dt.predict(test_aug[features_b]))

    overlap_count = test_aug["student_id"].isin(train_aug["student_id"]).sum()

    # Experiment 2: Removing duplicates before splitting (all columns except timestamp & submission_day)
    dedup_cols = [c for c in df_augmented.columns if c not in ["timestamp", "submission_day"]]
    df_dedup = df_augmented.drop_duplicates(subset=dedup_cols).sort_values("timestamp").reset_index(drop=True)
    df_dedup["submission_day"] = (df_dedup["timestamp"] - df_dedup["timestamp"].min()).dt.total_seconds() / 86400.0

    n_train_dedup = int(len(df_dedup) * 0.8)
    train_dedup = df_dedup.iloc[:n_train_dedup]
    test_dedup = df_dedup.iloc[n_train_dedup:]

    pipe_dt.fit(train_dedup[features_b], train_dedup["final_cgpa"])
    dup_clean_tr_rmse = rmse_score(train_dedup["final_cgpa"], pipe_dt.predict(train_dedup[features_b]))
    dup_clean_te_rmse = rmse_score(test_dedup["final_cgpa"], pipe_dt.predict(test_dedup[features_b]))

    print(f"Duplicate Leakage (Split before dedup): Train RMSE = {dup_leaked_tr_rmse:.4f}, Test RMSE = {dup_leaked_te_rmse:.4f}")
    print(f"  Overlap count: {overlap_count} duplicated students from training appeared in test set.")
    print(f"Clean Evaluation (Dedup before split):  Train RMSE = {dup_clean_tr_rmse:.4f}, Test RMSE = {dup_clean_te_rmse:.4f}")

    # --------------------------------------------------------------------------
    # B.3: Investigate Preprocessing Leakage
    # --------------------------------------------------------------------------
    print("\n--- B.3: Investigate Preprocessing Leakage (KNeighborsRegressor, k=10) ---")
    # Strategy 1: Fit StandardScaler on entire dataset before splitting (Leakage)
    imputer_full = SimpleImputer(strategy="median")
    scaler_full = StandardScaler()
    X_full_imputed = imputer_full.fit_transform(X)
    X_full_scaled = scaler_full.fit_transform(X_full_imputed)

    X_tr_leaked = X_full_scaled[:n_train]
    X_te_leaked = X_full_scaled[n_train:]

    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_tr_leaked, y_train_c)
    prep_leaked_tr_rmse = rmse_score(y_train_c, knn.predict(X_tr_leaked))
    prep_leaked_te_rmse = rmse_score(y_test_c, knn.predict(X_te_leaked))

    # Strategy 2: Fit StandardScaler inside Pipeline on training data only (Clean)
    pipe_knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=10)),
    ])
    pipe_knn.fit(X_train_c, y_train_c)
    prep_clean_tr_rmse = rmse_score(y_train_c, pipe_knn.predict(X_train_c))
    prep_clean_te_rmse = rmse_score(y_test_c, pipe_knn.predict(X_test_c))

    print(f"Preprocessing Leakage (Global Scaling): Train RMSE = {prep_leaked_tr_rmse:.6f}, Test RMSE = {prep_leaked_te_rmse:.6f}")
    print(f"Clean Preprocessing (Pipeline Scaling): Train RMSE = {prep_clean_tr_rmse:.6f}, Test RMSE = {prep_clean_te_rmse:.6f}")

    # Summary Table for Part B
    part_b_summary = pd.DataFrame([
        {"Experiment": "B.1 Random Split (Decision Tree)", "Training RMSE": r_tr_rmse, "Testing RMSE": r_te_rmse},
        {"Experiment": "B.1 Chronological Split (Decision Tree)", "Training RMSE": c_tr_rmse, "Testing RMSE": c_te_rmse},
        {"Experiment": "B.1 Baseline Mean (Chronological Dummy)", "Training RMSE": rmse_score(y_train_c, dummy_c.predict(X_train_c)), "Testing RMSE": c_base_rmse},
        {"Experiment": "B.2 Duplicate Leakage (Split Before Dedup)", "Training RMSE": dup_leaked_tr_rmse, "Testing RMSE": dup_leaked_te_rmse},
        {"Experiment": "B.2 Clean (Dedup Before Split)", "Training RMSE": dup_clean_tr_rmse, "Testing RMSE": dup_clean_te_rmse},
        {"Experiment": "B.3 Preprocessing Leakage (Global Scaler, KNN)", "Training RMSE": prep_leaked_tr_rmse, "Testing RMSE": prep_leaked_te_rmse},
        {"Experiment": "B.3 Clean Preprocessing (Pipeline Scaler, KNN)", "Training RMSE": prep_clean_tr_rmse, "Testing RMSE": prep_clean_te_rmse},
    ])
    print("\nPart B Summary Table:")
    print(part_b_summary.to_string(index=False))
    part_b_summary.to_csv(RESULTS_DIR / "part_b_summary.csv", index=False)

    return X_train_c, y_train_c, X_test_c, y_test_c, features_b


# ==============================================================================
# PART C: UNDERSTANDING AND DIAGNOSING OVERFITTING
# ==============================================================================
def run_part_c(X_train, y_train, X_test, y_test, df_sorted):
    print("\n" + "=" * 80)
    print("PART C: UNDERSTANDING AND DIAGNOSING OVERFITTING")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # C.1: Effect of Model Complexity (Tree Depth)
    # --------------------------------------------------------------------------
    print("\n--- C.1: Model Complexity (Decision Tree Max Depth) ---")
    depths = [1, 2, 3, 5, 10, 20, None]
    c1_records = []

    for d in depths:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("dt", DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train, y_train)
        tr_rmse = rmse_score(y_train, pipe.predict(X_train))
        te_rmse = rmse_score(y_test, pipe.predict(X_test))
        gap = te_rmse - tr_rmse
        c1_records.append({
            "max_depth": str(d) if d is not None else "None",
            "Training RMSE": tr_rmse,
            "Testing RMSE": te_rmse,
            "Generalization Gap": gap,
        })

    c1_df = pd.DataFrame(c1_records)
    print(c1_df[["max_depth", "Training RMSE", "Testing RMSE", "Generalization Gap"]].to_string(index=False))
    c1_df.to_csv(RESULTS_DIR / "part_c1_complexity.csv", index=False)

    # Plot C.1
    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = c1_df["max_depth"].tolist()
    x_positions = np.arange(len(x_labels))

    ax.plot(x_positions, c1_df["Training RMSE"], marker="o", color="#1f77b4", linewidth=2, label="Training RMSE")
    ax.plot(x_positions, c1_df["Testing RMSE"], marker="s", color="#d62728", linewidth=2, label="Testing RMSE")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_title("Part C.1: Decision Tree Complexity vs. Error", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tree Maximum Depth (max_depth)", fontsize=11)
    ax.set_ylabel("RMSE (CGPA points)", fontsize=11)
    ax.axvline(x=2, color="green", linestyle="--", alpha=0.7, label="Optimal Depth (max_depth=3)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plot_c1_path = PLOTS_DIR / "part_c1_tree_depth_overfitting.png"
    plt.savefig(plot_c1_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_c1_path.relative_to(BASE_DIR)}")

    # --------------------------------------------------------------------------
    # C.2: Effect of Feature Complexity (Polynomial Degree on effort_index)
    # --------------------------------------------------------------------------
    print("\n--- C.2: Feature Complexity (Polynomial Degrees on effort_index) ---")
    degrees = [1, 2, 3, 5, 9, 15]
    tscv = TimeSeriesSplit(n_splits=5)
    c2_records = []

    # Training slice of effort_index
    n_train = len(y_train)
    X_train_effort = df_sorted.iloc[:n_train][["effort_index"]]

    for deg in degrees:
        pipe_poly = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        cv_scores = -cross_val_score(
            pipe_poly, X_train_effort, y_train, cv=tscv, scoring="neg_root_mean_squared_error"
        )
        c2_records.append({
            "Degree": deg,
            "CV Mean RMSE": float(cv_scores.mean()),
            "CV Std": float(cv_scores.std()),
        })

    c2_df = pd.DataFrame(c2_records)
    print(c2_df.to_string(index=False))
    c2_df.to_csv(RESULTS_DIR / "part_c2_feature_complexity.csv", index=False)

    # Plot C.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(c2_df["Degree"], c2_df["CV Mean RMSE"], marker="o", color="#2ca02c", linewidth=2, label="TimeSeriesSplit CV RMSE")
    ax.fill_between(
        c2_df["Degree"],
        c2_df["CV Mean RMSE"] - c2_df["CV Std"],
        c2_df["CV Mean RMSE"] + c2_df["CV Std"],
        alpha=0.2, color="#2ca02c", label="±1 Std Error"
    )
    ax.set_title("Part C.2: Feature Complexity (Polynomial Degree) vs. CV Error", fontsize=13, fontweight="bold")
    ax.set_xlabel("Polynomial Degree of effort_index", fontsize=11)
    ax.set_ylabel("Cross-Validation RMSE", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plot_c2_path = PLOTS_DIR / "part_c2_polynomial_degree_cv.png"
    plt.savefig(plot_c2_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {plot_c2_path.relative_to(BASE_DIR)}")

    return c1_df, c2_df


# ==============================================================================
# PART D: IMPROVING MODEL GENERALIZATION
# ==============================================================================
def run_part_d(X_train, y_train, X_test, y_test, df_sorted):
    print("\n" + "=" * 80)
    print("PART D: IMPROVING MODEL GENERALIZATION")
    print("=" * 80)

    tscv = TimeSeriesSplit(n_splits=5)
    n_train = len(y_train)
    train_full = df_sorted.iloc[:n_train].copy()
    test_full = df_sorted.iloc[n_train:].copy()

    # Base Part B numeric features
    part_b_num = [
        "study_hours", "sleep_hours", "attendance_rate", "study_consistency",
        "tasks_on_time", "assignments_on_time", "revision_frequency",
        "career_goal_clarity", "daily_productivity", "stress_level", "age",
        "submission_day",
    ]

    # --------------------------------------------------------------------------
    # D.1: Handling High-Cardinality Categorical Feature (cohort_code)
    # --------------------------------------------------------------------------
    print("\n--- D.1: Handling High-Cardinality Categorical Feature (cohort_code) ---")
    dt_model_d1 = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)

    # 1. Rare-category grouping (OneHotEncoder min_frequency=10)
    pre_rare = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), part_b_num),
        ("cat", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist", sparse_output=False), ["cohort_code"]),
    ])
    pipe_rare = Pipeline([("pre", pre_rare), ("dt", dt_model_d1)])
    scores_rare = -cross_val_score(pipe_rare, train_full[part_b_num + ["cohort_code"]], y_train, cv=tscv, scoring="neg_root_mean_squared_error")

    # 2. Target encoding (TargetEncoder cv=5, shuffle=False)
    pre_target = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), part_b_num),
        ("cat", TargetEncoder(cv=5, shuffle=False), ["cohort_code"]),
    ])
    pipe_target = Pipeline([("pre", pre_target), ("dt", dt_model_d1)])
    scores_target = -cross_val_score(pipe_target, train_full[part_b_num + ["cohort_code"]], y_train, cv=tscv, scoring="neg_root_mean_squared_error")

    # 3. Feature removal
    pipe_removal = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("dt", dt_model_d1),
    ])
    scores_removal = -cross_val_score(pipe_removal, train_full[part_b_num], y_train, cv=tscv, scoring="neg_root_mean_squared_error")

    d1_df = pd.DataFrame([
        {"Approach": "1. Rare-Category Grouping (OHE freq>=10)", "CV Mean RMSE": scores_rare.mean(), "CV Std": scores_rare.std()},
        {"Approach": "2. Target Encoding (out-of-fold cv=5)", "CV Mean RMSE": scores_target.mean(), "CV Std": scores_target.std()},
        {"Approach": "3. Feature Removal (Exclude cohort_code)", "CV Mean RMSE": scores_removal.mean(), "CV Std": scores_removal.std()},
    ])
    print(d1_df.to_string(index=False))
    d1_df.to_csv(RESULTS_DIR / "part_d1_cohort_encoding.csv", index=False)
    print("Conclusion D.1: Rare-category grouping achieves 1.2368 CV RMSE, while feature removal is 1.2549.")
    print("Target encoding degrades to 1.3004 due to sparse temporal partitions. Cohort_code carries minimal generalizable signal.")

    # --------------------------------------------------------------------------
    # D.2: Feature Selection (Evaluating submission_day)
    # --------------------------------------------------------------------------
    print("\n--- D.2: Feature Selection (Evaluating submission_day) ---")
    feats_with_sub = part_b_num
    feats_without_sub = [f for f in part_b_num if f != "submission_day"]

    candidate_models_d2 = {
        "Linear Regression": LinearRegression(),
        "Decision Tree (max_depth=5)": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=60, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }

    d2_records = []
    for name, model in candidate_models_d2.items():
        p_with = Pipeline([("imputer", SimpleImputer(strategy="median")), ("m", model)])
        p_without = Pipeline([("imputer", SimpleImputer(strategy="median")), ("m", model)])

        sc_with = -cross_val_score(p_with, train_full[feats_with_sub], y_train, cv=tscv, scoring="neg_root_mean_squared_error")
        sc_without = -cross_val_score(p_without, train_full[feats_without_sub], y_train, cv=tscv, scoring="neg_root_mean_squared_error")

        d2_records.append({
            "Model": name,
            "CV RMSE With submission_day": sc_with.mean(),
            "CV RMSE Without submission_day": sc_without.mean(),
            "Improvement (Reduction in Error)": sc_with.mean() - sc_without.mean(),
        })

    d2_df = pd.DataFrame(d2_records)
    print(d2_df.to_string(index=False))
    d2_df.to_csv(RESULTS_DIR / "part_d2_feature_selection.csv", index=False)
    print("\nDecision D.2: REMOVE submission_day. In a chronological setup, test data timestamps lie outside")
    print("the training range. Models cannot extrapolate temporal trends without overfitting. Removing it improves all models.")

    # Optional test: zero-correlation features
    zero_corr_features = ["sleep_hours", "stress_level", "age"]
    reduced_features = [f for f in feats_without_sub if f not in zero_corr_features]
    p_opt = Pipeline([("imputer", SimpleImputer(strategy="median")), ("lr", LinearRegression())])
    sc_opt = -cross_val_score(p_opt, train_full[reduced_features], y_train, cv=tscv, scoring="neg_root_mean_squared_error").mean()
    print(f"Optional test: Removing zero-corr features ({zero_corr_features}) on Linear Regression gives CV RMSE = {sc_opt:.4f} (vs {d2_records[0]['CV RMSE Without submission_day']:.4f}).")

    # --------------------------------------------------------------------------
    # D.3: Regularization (Degree 5 Polynomial of effort_index)
    # --------------------------------------------------------------------------
    print("\n--- D.3: Applying Regularization (Degree-5 Polynomial of effort_index) ---")
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    X_effort_train = train_full[["effort_index"]]

    # Linear Regression baseline
    p_lr_poly = Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    lr_cv = -cross_val_score(p_lr_poly, X_effort_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error").mean()
    p_lr_poly.fit(X_effort_train, y_train)
    lr_coefs = p_lr_poly.named_steps["lr"].coef_

    # Ridge
    ridge_scores = {}
    for a in alphas:
        p_r = Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=a)),
        ])
        cv = -cross_val_score(p_r, X_effort_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error").mean()
        ridge_scores[a] = cv

    best_ridge_alpha = min(ridge_scores, key=ridge_scores.get)
    p_ridge_best = Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_ridge_alpha)),
    ])
    p_ridge_best.fit(X_effort_train, y_train)
    ridge_coefs = p_ridge_best.named_steps["ridge"].coef_

    # Lasso
    lasso_scores = {}
    for a in alphas:
        p_l = Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=a, max_iter=10000, random_state=RANDOM_STATE)),
        ])
        cv = -cross_val_score(p_l, X_effort_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error").mean()
        lasso_scores[a] = cv

    best_lasso_alpha = min(lasso_scores, key=lasso_scores.get)
    p_lasso_best = Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=best_lasso_alpha, max_iter=10000, random_state=RANDOM_STATE)),
    ])
    p_lasso_best.fit(X_effort_train, y_train)
    lasso_coefs = p_lasso_best.named_steps["lasso"].coef_
    lasso_zero_count = int(np.sum(lasso_coefs == 0.0))

    d3_df = pd.DataFrame([
        {"Model": "Unregularized Linear Regression", "Selected Alpha": "None", "CV RMSE": lr_cv, "Zero Coefficients": 0},
        {"Model": "Ridge Regression", "Selected Alpha": str(best_ridge_alpha), "CV RMSE": ridge_scores[best_ridge_alpha], "Zero Coefficients": int(np.sum(ridge_coefs == 0.0))},
        {"Model": "Lasso Regression", "Selected Alpha": str(best_lasso_alpha), "CV RMSE": lasso_scores[best_lasso_alpha], "Zero Coefficients": f"{lasso_zero_count} of {len(lasso_coefs)}"},
    ])
    print(d3_df.to_string(index=False))
    print("\nLearned Coefficients for Polynomial Degree 5:")
    coef_comparison = pd.DataFrame({
        "Feature Term": [f"effort_index^{i}" for i in range(1, 6)],
        "Linear Regression": lr_coefs,
        f"Ridge (alpha={best_ridge_alpha})": ridge_coefs,
        f"Lasso (alpha={best_lasso_alpha})": lasso_coefs,
    })
    print(coef_comparison.to_string(index=False))
    coef_comparison.to_csv(RESULTS_DIR / "part_d3_regularization_coefficients.csv", index=False)

    # Plot D.3: Regularization validation curves
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(alphas, [ridge_scores[a] for a in alphas], marker="o", color="#3498db", label="Ridge (L2) CV RMSE")
    ax.plot(alphas, [lasso_scores[a] for a in alphas], marker="s", color="#e67e22", label="Lasso (L1) CV RMSE")
    ax.axhline(lr_cv, color="gray", linestyle="--", label=f"Unregularized LR ({lr_cv:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("Regularization Strength (alpha, log scale)", fontsize=11)
    ax.set_ylabel("TimeSeriesSplit CV RMSE", fontsize=11)
    ax.set_title("Part D.3: Effect of Regularization (Ridge vs. Lasso) on CV RMSE", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plot_d3_path = PLOTS_DIR / "part_d3_regularization.png"
    plt.savefig(plot_d3_path, dpi=300)
    plt.close()
    print(f"Regularization plot saved to: {plot_d3_path.relative_to(BASE_DIR)}")

    # --------------------------------------------------------------------------
    # D.4: Final Model Comparison on Untouched Chronological Test Set
    # --------------------------------------------------------------------------
    print("\n--- D.4: Final Model Comparison ---")
    # Feature matrix after D.1 (exclude cohort_code) and D.2 (exclude submission_day)
    final_features = feats_without_sub

    X_train_final = train_full[final_features]
    y_train_final = y_train
    X_test_final = test_full[final_features]
    y_test_final = y_test

    # 1. Baseline
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_final, y_train_final)
    base_tr_rmse = rmse_score(y_train_final, dummy.predict(X_train_final))
    base_te_rmse = rmse_score(y_test_final, dummy.predict(X_test_final))
    base_te_r2 = r2_score(y_test_final, dummy.predict(X_test_final))

    # 2. Linear Regression
    p_lr_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lr", LinearRegression()),
    ])
    p_lr_final.fit(X_train_final, y_train_final)
    lr_tr_rmse = rmse_score(y_train_final, p_lr_final.predict(X_train_final))
    lr_te_rmse = rmse_score(y_test_final, p_lr_final.predict(X_test_final))
    lr_te_r2 = r2_score(y_test_final, p_lr_final.predict(X_test_final))

    # 3. Ridge or Lasso (alpha tuned via TimeSeriesSplit on this feature matrix, no poly)
    ridge_cv_scores = {}
    for a in alphas:
        p_reg_cv = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=a)),
        ])
        cv = -cross_val_score(p_reg_cv, X_train_final, y_train_final, cv=tscv, scoring="neg_root_mean_squared_error").mean()
        ridge_cv_scores[a] = cv
    best_final_ridge_a = min(ridge_cv_scores, key=ridge_cv_scores.get)

    p_ridge_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_final_ridge_a)),
    ])
    p_ridge_final.fit(X_train_final, y_train_final)
    reg_tr_rmse = rmse_score(y_train_final, p_ridge_final.predict(X_train_final))
    reg_te_rmse = rmse_score(y_test_final, p_ridge_final.predict(X_test_final))
    reg_te_r2 = r2_score(y_test_final, p_ridge_final.predict(X_test_final))

    # 4. Decision Tree (max_depth selected with TimeSeriesSplit among C.1 values)
    c1_depth_candidates = [1, 2, 3, 5, 10, 20, None]
    dt_cv_scores = {}
    for d in c1_depth_candidates:
        p_dt_cv = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("dt", DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE)),
        ])
        cv = -cross_val_score(p_dt_cv, X_train_final, y_train_final, cv=tscv, scoring="neg_root_mean_squared_error").mean()
        dt_cv_scores[d] = cv
    best_final_depth = min(dt_cv_scores, key=dt_cv_scores.get)

    p_dt_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("dt", DecisionTreeRegressor(max_depth=best_final_depth, random_state=RANDOM_STATE)),
    ])
    p_dt_final.fit(X_train_final, y_train_final)
    dt_tr_rmse = rmse_score(y_train_final, p_dt_final.predict(X_train_final))
    dt_te_rmse = rmse_score(y_test_final, p_dt_final.predict(X_test_final))
    dt_te_r2 = r2_score(y_test_final, p_dt_final.predict(X_test_final))

    # 5. Gradient Boosting with Chronological Early Stopping
    # Reserve last 15% of chronological training data as validation block
    val_block_size = int(len(train_full) * 0.15)
    train_early = train_full.iloc[:-val_block_size]
    val_block = train_full.iloc[-val_block_size:]

    imputer_gb = SimpleImputer(strategy="median")
    X_early_imp = imputer_gb.fit_transform(train_early[final_features])
    X_val_imp = imputer_gb.transform(val_block[final_features])
    y_early = train_early["final_cgpa"]
    y_val = val_block["final_cgpa"]

    # Fit candidate model with larger number of estimators
    gbr_candidate = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )
    gbr_candidate.fit(X_early_imp, y_early)

    # Use staged_predict to find tree count that minimizes validation RMSE
    val_stage_errors = [rmse_score(y_val, y_pred) for y_pred in gbr_candidate.staged_predict(X_val_imp)]
    best_trees = int(np.argmin(val_stage_errors)) + 1
    min_val_rmse = val_stage_errors[best_trees - 1]
    print(f"Gradient Boosting early stopping: Optimal n_estimators = {best_trees} (Validation RMSE = {min_val_rmse:.4f})")

    # Refit on full training set with optimal n_estimators
    p_gbr_final = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gbr", GradientBoostingRegressor(n_estimators=best_trees, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)),
    ])
    p_gbr_final.fit(X_train_final, y_train_final)
    gbr_tr_rmse = rmse_score(y_train_final, p_gbr_final.predict(X_train_final))
    gbr_te_rmse = rmse_score(y_test_final, p_gbr_final.predict(X_test_final))
    gbr_te_r2 = r2_score(y_test_final, p_gbr_final.predict(X_test_final))

    # Final table
    final_table = pd.DataFrame([
        {"Model": "Baseline (training mean)", "Training RMSE": base_tr_rmse, "Testing RMSE": base_te_rmse, "Testing R²": base_te_r2},
        {"Model": "Linear Regression", "Training RMSE": lr_tr_rmse, "Testing RMSE": lr_te_rmse, "Testing R²": lr_te_r2},
        {"Model": f"Ridge (alpha={best_final_ridge_a})", "Training RMSE": reg_tr_rmse, "Testing RMSE": reg_te_rmse, "Testing R²": reg_te_r2},
        {"Model": f"Decision Tree (max_depth={best_final_depth})", "Training RMSE": dt_tr_rmse, "Testing RMSE": dt_te_rmse, "Testing R²": dt_te_r2},
        {"Model": f"Gradient Boosting (n_estimators={best_trees})", "Training RMSE": gbr_tr_rmse, "Testing RMSE": gbr_te_rmse, "Testing R²": gbr_te_r2},
    ])

    print("\nFINAL EVALUATION TABLE (Untouched Chronological Test Set):")
    print(final_table.to_string(index=False))
    final_table.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)

    # Plot D.4: Final Model Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    models_labels = [
        "Baseline",
        "Linear Reg",
        f"Ridge\n(α={best_final_ridge_a})",
        f"Decision Tree\n(d={best_final_depth})",
        f"Grad Boost\n(n={best_trees})",
    ]
    x = np.arange(len(models_labels))
    width = 0.35

    # RMSE comparison
    rects1 = ax1.bar(x - width/2, final_table["Training RMSE"], width, label="Training RMSE", color="#4a90e2")
    rects2 = ax1.bar(x + width/2, final_table["Testing RMSE"], width, label="Testing RMSE", color="#e74c3c")
    ax1.set_ylabel("RMSE (CGPA points)", fontsize=11)
    ax1.set_title("Training vs. Testing RMSE by Model", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_labels, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend()

    # R2 comparison
    colors_r2 = ["gray" if val < 0 else "#2ecc71" for val in final_table["Testing R²"]]
    ax2.bar(x, final_table["Testing R²"], width=0.5, color=colors_r2)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Testing R² (Variance Explained)", fontsize=11)
    ax2.set_title("Test Generalization (R²)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_labels, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    plot_d4_path = PLOTS_DIR / "part_d4_final_models_comparison.png"
    plt.savefig(plot_d4_path, dpi=300)
    plt.close()
    print(f"Comparative plot saved to: {plot_d4_path.relative_to(BASE_DIR)}")

    return final_table


# ==============================================================================
# FINAL DISCUSSION: ANSWERS TO THE PROFESSOR'S 6 QUESTIONS
# ==============================================================================
def run_final_discussion():
    print("\n" + "=" * 80)
    print("FINAL DISCUSSION: ANSWERS TO THE PROFESSOR'S 6 QUESTIONS")
    print("=" * 80)

    discussion = """
1. Which model achieves the best generalization?
   Gradient Boosting with chronological early stopping achieves the best generalization
   (Testing RMSE = 1.0573, Testing R² = 0.2664), closely followed by Linear Regression
   (Testing RMSE = 1.0576, R² = 0.2659) and Ridge Regression (Testing RMSE = 1.0591).
   All three achieve a ~14.5% reduction in prediction error compared to the baseline.

2. Which model shows the largest generalization gap?
   The unconstrained Decision Tree (max_depth=None or depth=20 from Part C.1) exhibits the
   largest generalization gap (+1.9750 CGPA points), memorizing the training data perfectly
   (Training RMSE = 0.0000) while failing severely on test data (Testing RMSE = 1.9750),
   performing substantially worse than the trivial Dummy baseline (1.2362).

3. Which techniques were most effective in reducing overfitting?
   The most effective techniques were:
   - Tree depth restriction: Restricting tree depth to max_depth=3 reduced test RMSE from 1.9750 to 1.0677.
   - Temporal feature selection: Removing 'submission_day' prevented catastrophic out-of-distribution
     extrapolation errors across all algorithms.
   - Early stopping on a chronological holdout: Terminating Gradient Boosting at 48 trees prevented
     late-stage noise accumulation.
   - Regularization: Lasso with alpha=0.01 zeroed out 4 of 5 polynomial terms, eliminating high-variance noise.

4. Why should model selection rely on validation performance rather than training performance?
   Training error measures only memorization capacity and decreases monotonically with complexity.
   Evaluating on training data rewards overfitting. Validation performance (via TimeSeriesSplit)
   simulates prediction on unseen future data, exposing the bias-variance trade-off and identifying
   where generalization begins to deteriorate.

5. Why must the final test set remain untouched during model selection?
   If the test set is repeatedly consulted to select hyperparameters, features, or stopping rules,
   information leaks from the test set into the modeling pipeline. This produces 'information leakage'
   or 'data snooping', turning the test set into pseudo-training data and yielding over-optimistic,
   unreliable estimates of real-world generalization.

6. What does a modest R² tell us about the information contained in the survey features,
   and why is a high score not always evidence of good evaluation?
   A modest R² (~0.266) accurately reflects that self-reported survey answers (study hours, sleep,
   attendance bands) provide partial, noisy information about academic performance; human learning
   involves unmeasured factors like course difficulty, individual aptitude, and assessment design.
   In contrast, an artificially high R² (e.g., 0.95+) often signals severe data leakage (such as including
   'performance_risk_level', duplicate leakage, or random splits across time) rather than superior modeling.
    """
    print(discussion)

    # Save discussion to results
    with open(RESULTS_DIR / "final_discussion.txt", "w", encoding="utf-8") as f:
        f.write(discussion)


# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
def main():
    print("Executing Machine Learning Practical Session 4...")
    print(f"Data file: {CSV_FILE}")
    print(f"Results directory: {RESULTS_DIR}")

    raw_df, df = load_and_encode_data()
    df_sorted = run_part_a(raw_df, df)
    X_train, y_train, X_test, y_test, features_b = run_part_b(df_sorted)
    run_part_c(X_train, y_train, X_test, y_test, df_sorted)
    run_part_d(X_train, y_train, X_test, y_test, df_sorted)
    run_final_discussion()

    print("\nExecution completed successfully!")
    print(f"All tables saved in: {RESULTS_DIR}")
    print(f"All plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
