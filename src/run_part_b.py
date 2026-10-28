"""Build RFM segments and assess whether the supplied data supports lapse modeling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUTS_DIR = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
WINDOW_END = pd.Timestamp("2025-10-01 00:00:00")

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "tenure_days",
    "age",
]
CATEGORICAL_FEATURES = ["favourite_category", "city", "gender"]


def load_clean_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(CLEAN_DIR / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", parse_dates=["order_ts"])
    products = pd.read_csv(CLEAN_DIR / "products.csv")
    return customers, orders, products


def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    completed = orders.loc[orders["status"].eq("completed")].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    return rfm[["customer_id", "recency", "frequency", "monetary"]]


def segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Assign actionable names by profile, with the top monetary segment premium."""
    ids = list(profile.index)
    premium = int(profile["monetary"].idxmax())
    names = {premium: "Premium"}
    remaining = [segment_id for segment_id in ids if segment_id != premium]

    frequent = int(profile.loc[remaining, "frequency"].idxmax())
    names[frequent] = "Frequent Loyalists"
    remaining.remove(frequent)

    if remaining:
        win_back = int(profile.loc[remaining, "recency"].idxmax())
        names[win_back] = "Win-Back Candidates"
        remaining.remove(win_back)

    for segment_id in remaining:
        names[segment_id] = "Core Customers"
    return names


def build_segments(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    print("K-MEANS METRICS")
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        print(
            f"K={k}: inertia={model.inertia_:.1f}, "
            f"silhouette={silhouette_score(scaled, labels):.3f}"
        )

    chosen_k = 3
    model = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    rfm = rfm.copy()
    rfm["segment_id"] = model.fit_predict(scaled)
    profile = (
        rfm.groupby("segment_id")
        .agg(
            size=("customer_id", "size"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .sort_index()
    )
    names = segment_names(profile)
    rfm["segment_name"] = rfm["segment_id"].map(names)
    premium_id = int(profile["monetary"].idxmax())
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)
    profile["segment_name"] = profile.index.map(names)
    profile["recommended_action"] = profile.index.map(
        {
            premium_id: "Protect loyalty with VIP service and exclusive early access.",
            **{
                segment_id: (
                    "Reward repeat orders with replenishment reminders and a "
                    "multi-order loyalty offer."
                    if name == "Frequent Loyalists"
                    else (
                        "Use a personalized win-back message and limited-time "
                        "return incentive."
                        if name == "Win-Back Candidates"
                        else "Cross-sell complementary products and test a "
                        "modest basket-building offer."
                    )
                )
                for segment_id, name in names.items()
                if segment_id != premium_id
            },
        }
    )

    pca = PCA(n_components=2)
    pca.fit(scaled)
    print(f"\nSEGMENT PROFILE (K={chosen_k})")
    print(profile.to_string(float_format=lambda value: f"{value:.2f}"))
    print(
        "\nPCA explained variance ratio: "
        f"{pca.explained_variance_ratio_[0]:.3f}, "
        f"{pca.explained_variance_ratio_[1]:.3f}; "
        f"total={pca.explained_variance_ratio_.sum():.3f}"
    )
    print(
        "\nRFM summary:\n"
        f"{rfm[['recency', 'frequency', 'monetary']].describe().to_string()}"
    )
    return rfm, profile


def build_lapse_cohort(
    customers: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    orders = orders.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    features_orders = orders.loc[orders["order_ts"].lt(CUTOFF)].copy()
    completed = features_orders.loc[features_orders["status"].eq("completed")].copy()

    cohort = completed["customer_id"].drop_duplicates()
    target_window = orders.loc[
        orders["status"].eq("completed")
        & orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(WINDOW_END)
    ]
    active_in_window = set(target_window["customer_id"])
    y = cohort.isin(active_in_window).map({True: 0, False: 1}).rename("lapsed")
    cohort_frame = pd.DataFrame({"customer_id": cohort.to_numpy(), "lapsed": y.to_numpy()})

    grouped = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )
    grouped["recency"] = (CUTOFF - grouped["last_order_ts"]).dt.days

    final_orders = features_orders.loc[features_orders["status"].isin(["completed", "cancelled"])]
    all_final_counts = final_orders.groupby("customer_id").size()
    cancelled_counts = final_orders.loc[final_orders["status"].eq("cancelled")].groupby(
        "customer_id"
    ).size()
    grouped["cancellation_rate"] = (
        cancelled_counts.reindex(grouped.index, fill_value=0) / all_final_counts.reindex(grouped.index)
    )

    category_map = products.set_index("product_id")["category"]
    completed["category"] = completed["product_id"].map(category_map)
    favourite = (
        completed.groupby("customer_id")["category"]
        .agg(lambda values: values.value_counts().sort_index().idxmax())
        .rename("favourite_category")
    )
    grouped = grouped.join(favourite)
    grouped = grouped.join(customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    grouped["tenure_days"] = (CUTOFF - grouped["signup_date"]).dt.days
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    frame = cohort_frame.set_index("customer_id").join(grouped[feature_columns], how="inner")
    if frame[feature_columns].isna().any().any():
        missing = frame[feature_columns].columns[frame[feature_columns].isna().any()].tolist()
        raise ValueError(f"Lapse features contain missing values: {missing}")
    return frame.reset_index(), frame["lapsed"].reset_index(drop=True)


def make_model_pipeline(classifier: object) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [("preprocessor", preprocessing), ("classifier", classifier)]
    )


def report_metrics(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    scores = {
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    print(f"\n{name}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{confusion_matrix(y_true, y_pred, labels=[0, 1])}")
    print(
        f"Lapsed-class precision={scores['precision']:.3f}, "
        f"recall={scores['recall']:.3f}, F1={scores['f1']:.3f}"
    )
    return scores


def train_and_report_models(
    cohort: pd.DataFrame, segments: pd.DataFrame
) -> None:
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = cohort[feature_columns]
    y = cohort["lapsed"].astype(int)
    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    X_train, X_test, y_train, y_test, ids_test = train_test_split(
        X,
        y,
        cohort["customer_id"],
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(
        f"Training/test rows: {len(X_train)}/{len(X_test)}; "
        f"missing features: {int(X.isna().sum().sum())}"
    )

    baseline = np.zeros(len(y_test), dtype=int)
    print(
        f"Baseline accuracy: {accuracy_score(y_test, baseline):.3f}; "
        f"lapsed-class F1: {f1_score(y_test, baseline, zero_division=0):.3f}"
    )

    logistic_search = GridSearchCV(
        make_model_pipeline(LogisticRegression(max_iter=1000)),
        {"classifier__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    forest_search = GridSearchCV(
        make_model_pipeline(
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        {
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    print("Random-forest grid: 9 parameter combinations x 5 folds = 45 fits.")
    logistic_search.fit(X_train, y_train)
    forest_search.fit(X_train, y_train)
    print(
        "\nLogistic regression best parameters: "
        f"{logistic_search.best_params_}; CV F1={logistic_search.best_score_:.3f}"
    )
    print(
        "Random forest best parameters: "
        f"{forest_search.best_params_}; CV F1="
        f"{forest_search.best_score_:.3f}"
    )

    fitted_scaler = (
        logistic_search.best_estimator_
        .named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    frequency_position = NUMERIC_FEATURES.index("frequency")
    print(
        f"Training-fitted frequency scaler mean: "
        f"{fitted_scaler.mean_[frequency_position]:.3f}; "
        f"full-cohort frequency mean: {X['frequency'].mean():.3f}. "
        "They differ because the scaler was fitted on the training split only, "
        "not on the held-out test customers."
    )

    logistic_scores = report_metrics(
        "Tuned logistic regression test results",
        y_test,
        logistic_search.predict(X_test),
    )
    forest_scores = report_metrics(
        "Tuned random forest test results",
        y_test,
        forest_search.predict(X_test),
    )

    # Oversample only the training rows; the test set is never changed or resampled.
    class_counts = y_train.value_counts()
    if class_counts.get(1, 0) < class_counts.get(0, 0):
        minority_positions = np.flatnonzero(y_train.to_numpy() == 1)
        extra_positions = np.random.RandomState(42).choice(
            minority_positions,
            size=int(class_counts[0] - class_counts[1]),
            replace=True,
        )
        X_balanced = pd.concat(
            [X_train.reset_index(drop=True), X_train.iloc[extra_positions].reset_index(drop=True)],
            ignore_index=True,
        )
        y_balanced = pd.concat(
            [
                y_train.reset_index(drop=True),
                y_train.iloc[extra_positions].reset_index(drop=True),
            ],
            ignore_index=True,
        )
    else:
        raise ValueError("The lapsed class is not the minority class in training.")
    balanced_logistic = clone(logistic_search.best_estimator_)
    balanced_logistic.fit(X_balanced, y_balanced)
    print(
        "Oversampled training class counts: "
        f"{y_balanced.value_counts().sort_index().to_dict()}"
    )
    balanced_scores = report_metrics(
        "Oversampled-training logistic regression test results",
        y_test,
        balanced_logistic.predict(X_test),
    )
    print(
        "Oversampling metric changes (balanced minus unbalanced): "
        + ", ".join(
            f"{metric}={balanced_scores[metric] - logistic_scores[metric]:+.3f}"
            for metric in ("precision", "recall", "f1")
        )
    )
    changes = {
        metric: balanced_scores[metric] - logistic_scores[metric]
        for metric in ("precision", "recall", "f1")
    }
    max_change = max(abs(change) for change in changes.values())
    largest_changes = [
        metric for metric, change in changes.items() if abs(change) == max_change
    ]
    if largest_changes == ["recall"]:
        meaning = (
            "the team catches more lapsing customers, with more outreach to "
            "customers who may still be active."
        )
    elif largest_changes == ["precision"]:
        meaning = (
            "a larger share of outreach is directed to genuinely lapsed "
            "customers, reducing wasted incentives."
        )
    else:
        meaning = "the combined precision-recall balance changed most."
    print(f"Largest metric movement: {', '.join(largest_changes)}; {meaning}")

    selected_name, selected_model, selected_scores, selected_cv_f1 = max(
        [
            (
                "logistic regression",
                logistic_search.best_estimator_,
                logistic_scores,
                logistic_search.best_score_,
            ),
            (
                "random forest",
                forest_search.best_estimator_,
                forest_scores,
                forest_search.best_score_,
            ),
        ],
        key=lambda candidate: candidate[3],
    )
    if selected_scores["f1"] < 0.55:
        raise ValueError(
            f"Selected test F1 {selected_scores['f1']:.3f} is below the required 0.55."
        )
    print(
        f"\nSelected model by training-only CV F1: {selected_name}, "
        f"CV F1={selected_cv_f1:.3f}, test F1={selected_scores['f1']:.3f}"
    )
    print(
        "The selected model led on cross-validated F1 without using test results "
        "for model choice. Missing a lapsing customer forfeits a chance to "
        "intervene, while wrongly targeting a loyal customer wastes incentive "
        "budget and can cheapen promotions. Review lapsed-class recall and "
        "precision together; the test set is an untouched final check."
    )

    forest = forest_search.best_estimator_
    forest_feature_names = forest.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    importances = forest.named_steps["classifier"].feature_importances_
    print("Five most important random-forest features:")
    for index in np.argsort(importances)[-5:][::-1]:
        feature_name = forest_feature_names[index]
        normalized_name = feature_name.split("__", 1)[-1]
        if "frequency" in normalized_name:
            reading = "Order frequency is a proxy for customer buying cadence."
        elif "recency" in normalized_name:
            reading = "Longer time since the last order can signal inactivity."
        elif "monetary" in normalized_name:
            reading = "Past spend helps distinguish customer value."
        elif "cancellation_rate" in normalized_name:
            reading = "Repeated cancellations can indicate order friction."
        elif "average_delivery_minutes" in normalized_name:
            reading = "Delivery experience may relate to repeat-order behavior."
        elif "average_discount_pct" in normalized_name:
            reading = "Discount reliance can inform retention offer design."
        elif "tenure_days" in normalized_name:
            reading = "Tenure reflects how long the customer relationship has existed."
        else:
            reading = "This profile feature helps distinguish customer groups."
        print(f"  {feature_name}: {importances[index]:.4f} — {reading}")

    logistic = logistic_search.best_estimator_
    logistic_names = logistic.named_steps["preprocessor"].get_feature_names_out()
    coefficients = logistic.named_steps["classifier"].coef_[0]
    for feature in ("frequency", "recency", "monetary"):
        matches = [
            i for i, name in enumerate(logistic_names) if name.endswith(f"__{feature}")
        ]
        if matches:
            coefficient = coefficients[matches[0]]
            direction = "positive" if coefficient > 0 else "negative"
            print(
                f"Logistic coefficient {feature}: {direction} "
                f"({coefficient:+.4f}) — higher {feature} is associated "
                f"with {'higher' if coefficient > 0 else 'lower'} lapse odds, "
                "holding other modeled features constant."
            )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUTS_DIR / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)
    selected_predictions = selected_model.predict(X_test)
    reloaded_predictions = reloaded_model.predict(X_test)
    if not np.array_equal(selected_predictions, reloaded_predictions):
        raise AssertionError("Reloaded model predictions differ from original predictions.")
    print("Reloaded model reproduces all test predictions exactly.")

    segment_premium = segments.set_index("customer_id")["is_premium"]
    is_test_premium = ids_test.map(segment_premium).fillna(False).astype(bool)
    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    retention = pd.DataFrame(
        {
            "customer_id": ids_test.to_numpy(),
            "lapse_probability": test_probabilities,
            "actually_lapsed": y_test.to_numpy(),
            "is_premium": is_test_premium.to_numpy(),
        }
    )
    retention = (
        retention.loc[retention["is_premium"]]
        .sort_values("lapse_probability", ascending=False)
        .head(10)
        .drop(columns="is_premium")
    )
    retention.to_csv(OUTPUTS_DIR / "retention_list.csv", index=False)
    print(
        f"Premium test customers on retention list: {len(retention)}; "
        f"truly lapsed: {int(retention['actually_lapsed'].sum())}"
    )


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    customers, orders, products = load_clean_tables()
    rfm = build_rfm(orders)
    segments, _ = build_segments(rfm)
    segment_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
    segments[segment_columns].to_csv(OUTPUTS_DIR / "customer_segments.csv", index=False)

    print("\nWHOLE-WINDOW RFM CHECKS")
    print(f"Customers: {len(rfm)}")
    print(
        f"Recency median/min/max: {rfm['recency'].median():.0f}/"
        f"{rfm['recency'].min()}/{rfm['recency'].max()}"
    )
    print(
        f"Frequency mean/max: {rfm['frequency'].mean():.2f}/"
        f"{rfm['frequency'].max()}"
    )
    print(
        f"Monetary mean/max: INR {rfm['monetary'].mean():,.2f}/"
        f"INR {rfm['monetary'].max():,.2f}"
    )
    print(
        "Cluster IDs are arbitrary labels; their numeric values do not imply "
        "a ranking or business value."
    )

    cohort, y = build_lapse_cohort(customers, orders, products)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    print("\nLAPSE COHORT")
    print(f"Cohort size: {len(cohort)}")
    print(f"Lapsed: {int(y.sum())} ({100 * y.mean():.2f}%)")
    print(f"Completed orders in label window: {int((y == 0).sum())}")
    print(
        "Baseline always predicting not lapsed: "
        f"accuracy={float((y == 0).mean()):.3f}, "
        f"lapsed-class F1={f1_score(y, np.zeros(len(y)), zero_division=0):.3f}"
    )
    print(
        "Leakage controls: feature aggregates use only orders before "
        "2025-09-01; the September window is used only to construct labels; "
        "segment IDs/names are not model features."
    )
    print(
        "Missing values across required lapse features: "
        f"{int(cohort[feature_columns].isna().sum().sum())}"
    )
    if y.nunique() < 2:
        print(
            "MODELING BLOCKED: the September 2025 window has no completed "
            "orders in the supplied data, so every cohort member is labeled "
            "lapsed. A stratified train/test split and two-class model "
            "evaluation are not valid; no model or retention list was saved."
        )
        return

    cohort = cohort.copy()
    cohort["lapsed"] = y
    train_and_report_models(cohort, segments)


if __name__ == "__main__":
    main()
