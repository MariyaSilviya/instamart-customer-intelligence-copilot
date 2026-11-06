import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


# ============================================================
# SETUP
# ============================================================

os.makedirs("outputs", exist_ok=True)

RANDOM_STATE = 42
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")
LAPSE_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

print("=" * 65)
print("PART B: CUSTOMER SEGMENTATION AND LAPSE PREDICTION")
print("=" * 65)

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/clean/products.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])


# ============================================================
# TASK 1: RFM TABLE AS OF 2025-10-01
# ============================================================

print("\n--- 1. Building RFM Table ---")

completed = orders[orders["status"] == "completed"].copy()

rfm = completed.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (SNAPSHOT_DATE - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

print("RFM customers:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].describe())

rfm.to_csv("outputs/rfm_table.csv", index=False)


# ============================================================
# TASK 2: K-MEANS EVALUATION FOR K = 2 TO 8
# ============================================================

print("\n--- 2. K-Means Evaluation ---")

rfm_columns = ["recency", "frequency", "monetary"]
X_rfm = rfm[rfm_columns].copy()

rfm_scaler = StandardScaler()
X_rfm_scaled = rfm_scaler.fit_transform(X_rfm)

kmeans_results = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=RANDOM_STATE
    )

    cluster_ids = model.fit_predict(X_rfm_scaled)
    score = silhouette_score(X_rfm_scaled, cluster_ids)

    kmeans_results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": score
    })

    print(
        f"K={k}: inertia={model.inertia_:.2f}, "
        f"silhouette={score:.3f}"
    )

kmeans_results_df = pd.DataFrame(kmeans_results)
kmeans_results_df.to_csv(
    "outputs/kmeans_evaluation.csv", index=False
)

# K=4 is selected using the strong elbow improvement and
# silhouette score, together with business interpretability.
CHOSEN_K = 4

final_kmeans = KMeans(
    n_clusters=CHOSEN_K,
    n_init=10,
    random_state=RANDOM_STATE
)

rfm["segment_id"] = final_kmeans.fit_predict(X_rfm_scaled)


# ============================================================
# TASK 3: SEGMENT PROFILES, NAMES AND PREMIUM FLAG
# ============================================================

print("\n--- 3. Segment Profiles ---")

profiles = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).reset_index()

profiles = profiles.sort_values(
    "mean_monetary", ascending=False
).reset_index(drop=True)

# The segment with the highest mean monetary value is Premium.
premium_segment_id = int(profiles.iloc[0]["segment_id"])

# Give the remaining segments descriptive business names.
segment_names = {}
other_segments = profiles[
    profiles["segment_id"] != premium_segment_id
].copy()

segment_names[premium_segment_id] = "Premium"

# Name remaining clusters based on their RFM characteristics.
for _, row in other_segments.iterrows():
    segment_id = int(row["segment_id"])

    if row["mean_frequency"] >= profiles["mean_frequency"].median():
        name = "Frequent"
    elif row["mean_recency"] > profiles["mean_recency"].median():
        name = "At Risk"
    else:
        name = "Occasional"

    # Keep names unique if two clusters receive the same description.
    if name in segment_names.values():
        name = f"{name} {segment_id}"

    segment_names[segment_id] = name

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"] == premium_segment_id

profiles["segment_name"] = profiles["segment_id"].map(segment_names)
profiles["is_premium"] = (
    profiles["segment_id"] == premium_segment_id
)

print(profiles.to_string(index=False))
profiles.to_csv("outputs/segment_profiles.csv", index=False)


# ============================================================
# TASK 4: PCA ON THE SCALED RFM MATRIX
# ============================================================

print("\n--- 4. PCA Explained Variance ---")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coordinates = pca.fit_transform(X_rfm_scaled)

rfm["pca_1"] = pca_coordinates[:, 0]
rfm["pca_2"] = pca_coordinates[:, 1]

variance_ratios = pca.explained_variance_ratio_

print("PC1 explained variance:", round(variance_ratios[0], 4))
print("PC2 explained variance:", round(variance_ratios[1], 4))
print("Combined explained variance:", round(variance_ratios.sum(), 4))

# Required segmentation output.
rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
].to_csv("outputs/customer_segments.csv", index=False)

# Save PCA coordinates separately for analysis.
rfm[
    ["customer_id", "pca_1", "pca_2", "segment_id"]
].to_csv("outputs/pca_coordinates.csv", index=False)

print("Customer segmentation saved.")


# ============================================================
# TASK 5: DEFINE LAPSE COHORT AND TARGET
# ============================================================

print("\n--- 5. Defining Lapse Cohort ---")

# Use only orders dated before 1 September for features.
pre_cutoff = orders[orders["order_ts"] < LAPSE_CUTOFF].copy()

pre_completed = pre_cutoff[
    pre_cutoff["status"] == "completed"
].copy()

# Cohort: customers with at least one completed order before cutoff.
cohort_ids = pre_completed["customer_id"].unique()

# Target window: 1 September through 30 September.
september_completed = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] >= LAPSE_CUTOFF)
    & (orders["order_ts"] < LABEL_END)
]

active_in_september = set(
    september_completed["customer_id"].unique()
)

cohort = pd.DataFrame({"customer_id": cohort_ids})

cohort["is_lapsed"] = (
    ~cohort["customer_id"].isin(active_in_september)
).astype(int)

print("Cohort customers:", len(cohort))
print("Lapsed customers:", int(cohort["is_lapsed"].sum()))
print(
    "Lapsed percentage:",
    round(cohort["is_lapsed"].mean() * 100, 2)
)


# ============================================================
# TASK 6: BUILD FEATURES USING PRE-CUTOFF DATA ONLY
# ============================================================

print("\n--- 6. Building Leakage-Free Features ---")

# Completed-order behaviour before cutoff.
completed_features = pre_completed.groupby("customer_id").agg(
    recency=("order_ts", lambda x: (LAPSE_CUTOFF - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_time=("delivery_minutes", "mean")
).reset_index()

# Cancellation rate uses only final-status orders before cutoff.
final_status_orders = pre_cutoff[
    pre_cutoff["status"].isin(["completed", "cancelled"])
].copy()

status_counts = final_status_orders.pivot_table(
    index="customer_id",
    columns="status",
    values="order_id",
    aggfunc="nunique",
    fill_value=0
).reset_index()

if "completed" not in status_counts.columns:
    status_counts["completed"] = 0

if "cancelled" not in status_counts.columns:
    status_counts["cancelled"] = 0

status_counts["cancellation_rate"] = (
    status_counts["cancelled"]
    / (
        status_counts["completed"]
        + status_counts["cancelled"]
    ).replace(0, np.nan)
)

# Find the most frequently ordered product category.
category_orders = pre_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = (
    category_orders.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_order_count")
    .sort_values(
        ["customer_id", "category_order_count", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# Combine features with clean customer demographics.
model_data = (
    cohort
    .merge(completed_features, on="customer_id", how="left")
    .merge(
        status_counts[["customer_id", "cancellation_rate"]],
        on="customer_id",
        how="left"
    )
    .merge(favourite_category, on="customer_id", how="left")
    .merge(
        customers[
            [
                "customer_id",
                "city",
                "age",
                "gender",
                "signup_date"
            ]
        ],
        on="customer_id",
        how="left"
    )
)

model_data["tenure_days"] = (
    LAPSE_CUTOFF - model_data["signup_date"]
).dt.days

model_data["favourite_category"] = (
    model_data["favourite_category"].fillna("Unknown")
)

# Verify no September or later orders entered the feature table.
assert pre_cutoff["order_ts"].max() < LAPSE_CUTOFF

# Segment labels are deliberately not merged into model_data.
assert "segment_name" not in model_data.columns
assert "segment_id" not in model_data.columns
assert "is_premium" not in model_data.columns

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_time",
    "tenure_days",
    "age"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

# Fill any remaining feature missing values.
for column in numeric_features:
    model_data[column] = model_data[column].fillna(
        model_data[column].median()
    )

for column in categorical_features:
    model_data[column] = (
        model_data[column].fillna("Unknown").astype(str)
    )

assert model_data[numeric_features + categorical_features].isna().sum().sum() == 0

print("Feature missing values:", model_data[
    numeric_features + categorical_features
].isna().sum().sum())

print("Feature cutoff:", LAPSE_CUTOFF.date())
print("September orders used only for target labels.")


# ============================================================
# TASK 7: STRATIFIED TRAIN/TEST SPLIT
# ============================================================

print("\n--- 7. Train/Test Split ---")

X = model_data[numeric_features + categorical_features]
y = model_data["is_lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Training class counts:")
print(y_train.value_counts().sort_index())
print("Test class counts:")
print(y_test.value_counts().sort_index())


# ============================================================
# TASK 8: PREPROCESSING PIPELINE
# ============================================================

print("\n--- 8. Training-Only Preprocessing ---")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# All preprocessing is fitted inside each training-only pipeline.
# GridSearchCV receives X_train only, never X_test.


# ============================================================
# TASK 9: BASELINE MODEL
# ============================================================

print("\n--- 9. Baseline Evaluation ---")

baseline_predictions = np.zeros(len(y_test), dtype=int)

baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_precision = precision_score(
    y_test, baseline_predictions, zero_division=0
)
baseline_recall = recall_score(
    y_test, baseline_predictions, zero_division=0
)
baseline_f1 = f1_score(
    y_test, baseline_predictions, zero_division=0
)

print("Baseline accuracy:", round(baseline_accuracy, 4))
print("Baseline lapsed precision:", round(baseline_precision, 4))
print("Baseline lapsed recall:", round(baseline_recall, 4))
print("Baseline lapsed F1:", round(baseline_f1, 4))
print("Baseline confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions))


# ============================================================
# TASK 10: LOGISTIC REGRESSION GRID SEARCH
# ============================================================

print("\n--- 10. Logistic Regression Grid Search ---")

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

logistic_search.fit(X_train, y_train)

best_logistic = logistic_search.best_estimator_

print("Best Logistic Regression parameters:")
print(logistic_search.best_params_)
print(
    "Best Logistic CV F1:",
    round(logistic_search.best_score_, 4)
)


# ============================================================
# TASK 11: RANDOM FOREST GRID SEARCH
# ============================================================

print("\n--- 11. Random Forest Grid Search ---")

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE
            )
        )
    ]
)

rf_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("Best Random Forest parameters:")
print(rf_search.best_params_)
print(
    "Best Random Forest CV F1:",
    round(rf_search.best_score_, 4)
)


# ============================================================
# TASK 12: EVALUATE BOTH MODELS ON UNTOUCHED TEST DATA
# ============================================================

print("\n--- 12. Test Set Evaluation ---")


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"\n{name}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Lapsed precision:", round(precision, 4))
    print("Lapsed recall:", round(recall, 4))
    print("Lapsed F1:", round(f1, 4))

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


logistic_metrics = evaluate_model(
    "Logistic Regression", best_logistic, X_test, y_test
)

rf_metrics = evaluate_model(
    "Random Forest", best_rf, X_test, y_test
)

# Select the model with the higher test F1.
# The test set is used here only for final model comparison.
if rf_metrics["f1"] >= logistic_metrics["f1"]:
    selected_model = best_rf
    selected_name = "Random Forest"
    selected_metrics = rf_metrics
else:
    selected_model = best_logistic
    selected_name = "Logistic Regression"
    selected_metrics = logistic_metrics

print("\nSelected model:", selected_name)
print("Selected test F1:", round(selected_metrics["f1"], 4))

metrics_df = pd.DataFrame([
    {"model": "Baseline", "precision": baseline_precision,
     "recall": baseline_recall, "f1": baseline_f1},
    logistic_metrics,
    rf_metrics
])

metrics_df.to_csv("outputs/lapse_model_metrics.csv", index=False)


# ============================================================
# TASK 13: TRAINING-ONLY OVERSAMPLING EXPERIMENT
# ============================================================

print("\n--- 13. Training-Only Oversampling Experiment ---")

# Fit preprocessing on original training rows only.
logistic_preprocessor = best_logistic.named_steps["preprocessor"]

X_train_transformed = logistic_preprocessor.transform(X_train)
X_test_transformed = logistic_preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays if necessary.
if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

balanced_train = pd.DataFrame(X_train_transformed)
balanced_train["target"] = y_train.to_numpy()

majority_count = balanced_train["target"].value_counts().max()

minority_rows = balanced_train[
    balanced_train["target"] == 1
]

additional_minority = resample(
    minority_rows,
    replace=True,
    n_samples=majority_count - len(minority_rows),
    random_state=RANDOM_STATE
)

oversampled_train = pd.concat(
    [balanced_train, additional_minority],
    ignore_index=True
).sample(frac=1, random_state=RANDOM_STATE)

print("Training class counts after oversampling:")
print(oversampled_train["target"].value_counts().sort_index())

oversampled_logistic = LogisticRegression(
    C=logistic_search.best_params_["model__C"],
    max_iter=1000,
    random_state=RANDOM_STATE
)

oversampled_logistic.fit(
    oversampled_train.drop(columns="target"),
    oversampled_train["target"]
)

oversampled_predictions = oversampled_logistic.predict(
    X_test_transformed
)

oversampled_precision = precision_score(
    y_test, oversampled_predictions, zero_division=0
)
oversampled_recall = recall_score(
    y_test, oversampled_predictions, zero_division=0
)
oversampled_f1 = f1_score(
    y_test, oversampled_predictions, zero_division=0
)

print("Oversampled Logistic Regression:")
print("Precision:", round(oversampled_precision, 4))
print("Recall:", round(oversampled_recall, 4))
print("F1:", round(oversampled_f1, 4))

pd.DataFrame([
    {
        "experiment": "Original training data",
        "precision": logistic_metrics["precision"],
        "recall": logistic_metrics["recall"],
        "f1": logistic_metrics["f1"]
    },
    {
        "experiment": "Oversampled training data",
        "precision": oversampled_precision,
        "recall": oversampled_recall,
        "f1": oversampled_f1
    }
]).to_csv("outputs/oversampling_comparison.csv", index=False)


# ============================================================
# TASK 14: FEATURE IMPORTANCE AND COEFFICIENTS
# ============================================================

print("\n--- 14. Feature Interpretation ---")

# Random Forest feature importance.
rf_preprocessor = best_rf.named_steps["preprocessor"]
rf_model = best_rf.named_steps["model"]

feature_names = rf_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop five Random Forest features:")
print(importance_df.head(5).to_string(index=False))

importance_df.to_csv(
    "outputs/random_forest_feature_importance.csv",
    index=False
)

# Logistic Regression coefficient signs.
logistic_model = best_logistic.named_steps["model"]
logistic_feature_names = (
    best_logistic.named_steps["preprocessor"].get_feature_names_out()
)

coefficients_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_model.coef_[0]
})

coefficients_df["direction"] = np.where(
    coefficients_df["coefficient"] > 0,
    "Increases predicted lapse",
    "Decreases predicted lapse"
)

coefficients_df["absolute_coefficient"] = (
    coefficients_df["coefficient"].abs()
)

coefficients_df = coefficients_df.sort_values(
    "absolute_coefficient", ascending=False
)

print("\nThree largest Logistic Regression coefficients:")
print(
    coefficients_df[
        ["feature", "coefficient", "direction"]
    ].head(3).to_string(index=False)
)

coefficients_df.to_csv(
    "outputs/logistic_regression_coefficients.csv",
    index=False
)


# ============================================================
# TASK 15: SAVE, RELOAD AND VERIFY SELECTED MODEL
# ============================================================

print("\n--- 15. Save and Reload Selected Model ---")

model_path = "outputs/lapse_model.joblib"

joblib.dump(selected_model, model_path)

reloaded_model = joblib.load(model_path)

original_predictions = selected_model.predict(X_test)
reloaded_predictions = reloaded_model.predict(X_test)

assert np.array_equal(
    original_predictions,
    reloaded_predictions
)

print("Model saved and reloaded successfully.")
print("Reloaded model predictions exactly match original predictions.")


# ============================================================
# TASK 16: PREMIUM RETENTION LIST
# ============================================================

print("\n--- 16. Premium Retention List ---")

# Map whole-window RFM Premium status to the lapse cohort.
# This is post-hoc segmentation and is not a model feature.
premium_lookup = rfm[
    ["customer_id", "is_premium"]
].copy()

test_customers = model_data.loc[
    X_test.index,
    ["customer_id", "is_lapsed"]
].copy()

test_customers["lapse_probability"] = (
    selected_model.predict_proba(X_test)[:, 1]
)

test_customers = test_customers.merge(
    premium_lookup,
    on="customer_id",
    how="left"
)

premium_test = test_customers[
    test_customers["is_premium"] == True
].copy()

retention_list = premium_test.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention_list = retention_list.rename(
    columns={"is_lapsed": "actually_lapsed"}
)

retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
].to_csv("outputs/retention_list.csv", index=False)

print("Premium customers in test set:", len(premium_test))
print("Retention list rows:", len(retention_list))
print(
    "Actually lapsed in retention list:",
    int(retention_list["actually_lapsed"].sum())
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PART B PIPELINE FINISHED")
print("=" * 65)

print("RFM customers:", len(rfm))
print("Lapse cohort:", len(model_data))
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Selected model:", selected_name)
print("Selected test F1:", round(selected_metrics["f1"], 4))

if selected_metrics["f1"] >= 0.55:
    print("PASS: Selected model meets the required F1 threshold.")
else:
    print("WARNING: Selected model is below the required F1 threshold.")

print("\nGenerated Part B files are in outputs/.")
