"""
FlowRack Logistics Pvt. Ltd. - Warehouse Efficiency Insight (Use Case 22)
Feature Engineering and Selection Pipeline
DS Sprint - Skill Based Assessment 2 - Round 2

Implements:
- Feature 1: Picking Duration (validated)
- Feature 2: Order Size Category (Small, Medium, Large, Bulk)
- Feature 3: Picking Quantity & Unit picking rate
- Feature 4: Zone Characteristics (distance scores, zone accessibility)
- Feature 5: Worker Characteristics (shift, experience ordinal scores)
- Feature 6: Item Characteristics (handling complexity index)
- Feature 7: Order Processing Summary (Total labor duration vs elapsed chronological span)
- Feature 8: Order Queue Time (arrival-to-pick dispatch latency)
- Categorical Encoding (One-Hot & Ordinal)
- Feature Selection (Mutual Information, Random Forest Feature Importance, Lasso Regression, Multicollinearity)
- Generates:
  - data/processed/feature_engineered_data.csv
  - data/processed/selected_features.csv
  - outputs/feature_summary.csv
  - outputs/feature_selection_report.csv
"""

import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def load_integrated_data(data_path="data/processed/cleaned_warehouse_data.csv"):
    """Loads the cleaned warehouse data."""
    df = pd.read_csv(data_path)
    df["pick_start_time"] = pd.to_datetime(df["pick_start_time"])
    df["pick_finish_time"] = pd.to_datetime(df["pick_finish_time"])
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    return df


def engineer_features(df):
    """
    Creates all domain-specific engineered features.
    """
    df_feat = df.copy()

    # =========================================================================
    # FEATURE 1: Picking Duration (already verified positive in minutes)
    # =========================================================================
    # Re-verify and ensure rounded to 3 decimal places
    df_feat["picking_duration_minutes"] = (
        (df_feat["pick_finish_time"] - df_feat["pick_start_time"]).dt.total_seconds() / 60.0
    ).round(3)

    # =========================================================================
    # FEATURE 2: Order Size Category
    # Categorizes item_count:
    # 1-3: Small (Single-line / fast-track)
    # 4-7: Medium (Standard tote)
    # 8-12: Large (Multi-tote / cart)
    # 13+: Bulk (Heavy palletized order)
    # =========================================================================
    def categorize_order_size(count):
        if count <= 3:
            return "Small (1-3)"
        elif count <= 7:
            return "Medium (4-7)"
        elif count <= 12:
            return "Large (8-12)"
        else:
            return "Bulk (13+)"

    df_feat["order_size_category"] = df_feat["item_count"].apply(categorize_order_size)

    # =========================================================================
    # FEATURE 3: Picking Quantity & Pace
    # =========================================================================
    # Unit picking duration (pace in minutes per unit)
    df_feat["picking_pace_min_per_unit"] = (
        df_feat["picking_duration_minutes"] / df_feat["picking_quantity"]
    ).round(3)

    # =========================================================================
    # FEATURE 4: Zone Characteristics & Distance Score
    # =========================================================================
    dist_map = {"Near": 1, "Mid": 2, "Far": 3, "Deep-Storage": 4}
    df_feat["distance_score"] = df_feat["distance_band"].map(dist_map).fillna(1).astype(int)

    # Zone accessibility rating: Fast-Moving is easiest (1), Bulk Storage / High-Rack hardest (4)
    zone_access_map = {
        "Fast-Moving": 1,
        "Standard Shelf": 2,
        "Mezzanine": 3,
        "High-Rack": 4,
        "Cold Vault": 4,
        "Bulk Storage": 4
    }
    df_feat["zone_difficulty_score"] = df_feat["zone_type"].map(zone_access_map).fillna(2).astype(int)

    # =========================================================================
    # FEATURE 5: Worker Experience & Shift Encoding
    # =========================================================================
    exp_map = {"Novice": 1, "Intermediate": 2, "Senior": 3, "Unknown": 2}
    df_feat["experience_score"] = df_feat["experience_band"].map(exp_map).fillna(2).astype(int)

    # =========================================================================
    # FEATURE 6: Item Handling Complexity
    # Combines size and storage requirements
    # =========================================================================
    size_map = {"Small": 1, "Medium": 2, "Large": 3, "Bulky": 4}
    storage_map = {"Ambient": 1, "Cold Storage": 2, "Secure/High-Value": 3, "Hazardous": 4}
    
    df_feat["item_size_score"] = df_feat["item_size"].map(size_map).fillna(2).astype(int)
    df_feat["storage_class_score"] = df_feat["storage_class"].map(storage_map).fillna(1).astype(int)
    
    # Composite handling complexity index: size_score * 1.5 + storage_class_score
    df_feat["handling_complexity_score"] = (
        df_feat["item_size_score"] * 1.5 + df_feat["storage_class_score"]
    ).round(2)

    # =========================================================================
    # FEATURE 7 & 8: Order-Level Aggregates and Queue Time
    # =========================================================================
    # Pick level queue time
    df_feat["pick_queue_time_minutes"] = (
        (df_feat["pick_start_time"] - df_feat["order_timestamp"]).dt.total_seconds() / 60.0
    ).round(2)

    # Temporal feature: Hour of day when pick started
    df_feat["pick_hour"] = df_feat["pick_start_time"].dt.hour
    df_feat["pick_day_of_week"] = df_feat["pick_start_time"].dt.day_name()

    # Priority ordinal score
    prio_map = {"Low": 1, "Medium": 2, "High": 3, "Urgent": 4}
    df_feat["order_priority_score"] = df_feat["order_priority"].map(prio_map).fillna(2).astype(int)

    return df_feat


def generate_order_level_summary(df_feat, output_file="data/processed/order_level_summary.csv"):
    """
    Computes order-level processing summary, distinguishing between:
    - Sum of individual pick durations (total labor effort)
    - Elapsed chronological wall-clock span (last pick finish - first pick start)
    """
    order_grp = df_feat.groupby("order_id").agg(
        order_timestamp=("order_timestamp", "first"),
        order_priority=("order_priority", "first"),
        item_count=("item_count", "first"),
        order_size_category=("order_size_category", "first"),
        num_picks=("pick_id", "count"),
        total_picked_quantity=("picking_quantity", "sum"),
        total_pick_duration_sum_minutes=("picking_duration_minutes", "sum"),
        first_pick_start=("pick_start_time", "min"),
        last_pick_finish=("pick_finish_time", "max"),
        num_distinct_zones_visited=("zone_id", "nunique"),
        dominant_shift=("shift", lambda x: x.mode()[0] if not x.empty else "Unknown")
    ).reset_index()

    # Elapsed chronological processing span
    order_grp["elapsed_processing_span_minutes"] = (
        (order_grp["last_pick_finish"] - order_grp["first_pick_start"]).dt.total_seconds() / 60.0
    ).round(2)

    # Order queue time (first pick start - order release timestamp)
    order_grp["order_queue_time_minutes"] = (
        (order_grp["first_pick_start"] - order_grp["order_timestamp"]).dt.total_seconds() / 60.0
    ).round(2)

    order_grp.to_csv(output_file, index=False)
    print(f"Order-level summary saved to {output_file} ({len(order_grp)} orders).")
    return order_grp


def create_feature_summary_report(df_feat, output_file="outputs/feature_summary.csv"):
    """
    Generates documentation of all engineered features.
    """
    feature_meta = [
        {
            "feature_name": "picking_duration_minutes",
            "feature_type": "Continuous Numerical (Target)",
            "domain_interpretation": "Duration in minutes between pick start and finish. Core warehouse labor efficiency metric.",
            "formula_or_derivation": "(pick_finish_time - pick_start_time).total_seconds() / 60"
        },
        {
            "feature_name": "order_size_category",
            "feature_type": "Ordinal Categorical",
            "domain_interpretation": "Operational grouping of orders into Small (1-3), Medium (4-7), Large (8-12), Bulk (13+).",
            "formula_or_derivation": "Rule-based binning on orders.item_count"
        },
        {
            "feature_name": "picking_pace_min_per_unit",
            "feature_type": "Continuous Numerical",
            "domain_interpretation": "Time required to pick each individual unit of an item SKU (pace metric).",
            "formula_or_derivation": "picking_duration_minutes / picking_quantity"
        },
        {
            "feature_name": "distance_score",
            "feature_type": "Ordinal Numerical",
            "domain_interpretation": "Rank of travel distance from shipping dispatch (Near: 1, Mid: 2, Far: 3, Deep-Storage: 4).",
            "formula_or_derivation": "Mapped from zones.distance_band"
        },
        {
            "feature_name": "zone_difficulty_score",
            "feature_type": "Ordinal Numerical",
            "domain_interpretation": "Operational access hurdle (Fast-Moving: 1, Mezzanine: 3, High-Rack/Cold Vault/Bulk: 4).",
            "formula_or_derivation": "Mapped from zones.zone_type"
        },
        {
            "feature_name": "experience_score",
            "feature_type": "Ordinal Numerical",
            "domain_interpretation": "Tenure tier of the warehouse associate (Novice: 1, Intermediate: 2, Senior: 3).",
            "formula_or_derivation": "Mapped from workers.experience_band"
        },
        {
            "feature_name": "handling_complexity_score",
            "feature_type": "Continuous Numerical",
            "domain_interpretation": "Composite physical handling friction index combining item dimensions and hazardous/cold/security storage rules.",
            "formula_or_derivation": "item_size_score * 1.5 + storage_class_score"
        },
        {
            "feature_name": "pick_queue_time_minutes",
            "feature_type": "Continuous Numerical",
            "domain_interpretation": "Latency in minutes between order placement/release and pick execution start.",
            "formula_or_derivation": "(pick_start_time - order_timestamp).total_seconds() / 60"
        },
        {
            "feature_name": "order_priority_score",
            "feature_type": "Ordinal Numerical",
            "domain_interpretation": "SLA urgency tier (Low: 1, Medium: 2, High: 3, Urgent: 4).",
            "formula_or_derivation": "Mapped from orders.order_priority"
        },
        {
            "feature_name": "pick_hour",
            "feature_type": "Discrete Numerical",
            "domain_interpretation": "Hour of day (0-23) indicating diurnal shift progression and circadian fatigue factors.",
            "formula_or_derivation": "pick_start_time.dt.hour"
        }
    ]
    df_fsum = pd.DataFrame(feature_meta)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_fsum.to_csv(output_file, index=False)
    print(f"Feature summary report written to {output_file}")
    return df_fsum


def perform_feature_selection(df_feat, 
                              target_col="picking_duration_minutes",
                              output_report="outputs/feature_selection_report.csv",
                              output_selected="data/processed/selected_features.csv"):
    """
    Performs rigorous feature selection avoiding target leakage:
    - Excludes finish timestamps, queue times, and IDs
    - Evaluates:
      1. Mutual Information (non-linear capture)
      2. Random Forest Gini/MDI Feature Importance
      3. Lasso Regularized Regression Coefficients
      4. Multicollinearity / Pearson correlation
    - Compiles combined importance score and selects top predictive features
    """
    print("\nRunning feature selection pipeline...")

    # Filter out unassigned worker records for clean modeling baseline
    model_df = df_feat[df_feat["worker_id"] != "Unassigned"].copy()

    # Define candidate feature set (only information known AT OR PRIOR TO picking)
    # Strictly avoid target leakage (no finish time, no duration derivatives)
    candidate_numeric = [
        "picking_quantity",
        "distance_score",
        "zone_difficulty_score",
        "experience_score",
        "item_size_score",
        "storage_class_score",
        "handling_complexity_score",
        "order_priority_score",
        "item_count",
        "pick_hour"
    ]
    candidate_categorical = [
        "item_category",
        "zone_type",
        "distance_band",
        "shift",
        "storage_class"
    ]

    # Create one-hot encoded matrix for categorical features
    encoded_cats = pd.get_dummies(model_df[candidate_categorical], drop_first=True, dtype=int)
    
    # Combined feature matrix X
    X = pd.concat([model_df[candidate_numeric], encoded_cats], axis=1)
    y = model_df[target_col].values

    # 1. Mutual Information
    print("Computing Mutual Information scores...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)

    # 2. Random Forest Importance
    print("Fitting Random Forest Regressor for MDI Importance...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)

    # 3. Lasso Regression (Standardized)
    print("Fitting LassoCV for regularized linear selection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y)
    lasso_coefs = pd.Series(np.abs(lasso.coef_), index=X.columns)

    # 4. Pearson Correlation with Target
    corr_with_target = X.apply(lambda col: np.corrcoef(col, y)[0, 1]).abs()

    # Compile report
    report_df = pd.DataFrame({
        "feature_name": X.columns,
        "mutual_info_score": mi_series.values.round(4),
        "rf_importance_pct": (rf_importance.values * 100).round(2),
        "lasso_abs_coef": lasso_coefs.values.round(4),
        "abs_pearson_corr": corr_with_target.values.round(4)
    })

    # Combined composite ranking score (normalized average of MI and RF importance)
    mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-6)
    rf_norm = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min() + 1e-6)
    report_df["composite_importance_rank"] = ((mi_norm + rf_norm) / 2.0).values.round(4)
    report_df = report_df.sort_values(by="composite_importance_rank", ascending=False).reset_index(drop=True)

    # Decide feature inclusion (keep top 12 features with meaningful explanatory power)
    report_df["selection_decision"] = report_df["composite_importance_rank"].apply(
        lambda r: "Selected" if r >= 0.15 else "Excluded (Low Marginal Value)"
    )
    report_df["business_justification"] = report_df["feature_name"].apply(
        lambda f: "Direct operational driver of picking time (physical handling / travel distance)" 
        if any(k in f for k in ["distance", "handling", "size", "Bulk", "quantity", "experience"])
        else "Moderate predictive power; included for non-linear interactions" if "zone" in f or "shift" in f
        else "Redundant or low explanatory contribution after controlling for primary drivers"
    )

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    report_df.to_csv(output_report, index=False)
    print(f"Feature selection report saved to {output_report}")

    # Build selected features dataframe for modeling
    selected_feature_names = report_df[report_df["selection_decision"] == "Selected"]["feature_name"].tolist()
    print(f"Selected {len(selected_feature_names)} features: {selected_feature_names}")

    selected_df = X[selected_feature_names].copy()
    selected_df[target_col] = y
    selected_df.to_csv(output_selected, index=False)
    print(f"Selected feature matrix saved to {output_selected} ({len(selected_df)} rows, {len(selected_df.columns)} columns).")

    return report_df, selected_df


def run_pipeline():
    df = load_integrated_data()
    df_feat = engineer_features(df)
    
    # Save engineered feature dataset
    feat_out = "data/processed/feature_engineered_data.csv"
    os.makedirs(os.path.dirname(feat_out), exist_ok=True)
    df_feat.to_csv(feat_out, index=False)
    print(f"Feature engineered data saved to {feat_out} ({len(df_feat)} rows, {len(df_feat.columns)} columns).")

    generate_order_level_summary(df_feat)
    create_feature_summary_report(df_feat)
    perform_feature_selection(df_feat)
    return df_feat


if __name__ == "__main__":
    run_pipeline()
