"""
FlowRack Logistics Pvt. Ltd. - Warehouse Efficiency Insight (Use Case 22)
Model Performance & Benchmark Evaluation Pipeline

Evaluates multiple regression algorithms predicting picking duration:
1. Baseline (Dummy Regressor - Mean predictor)
2. Linear Regression (OLS)
3. Ridge Regression (L2 Regularization)
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. XGBoost Regressor

Calculates:
- Train and Test R2, MAE, RMSE, MAPE
- 5-Fold Cross-Validation R2 (Mean +/- Std)
- Generalization Error / Overfitting Gap
- Exports:
  - outputs/model_performance_report.csv
  - outputs/figures/model_comparison.png
  - outputs/figures/predicted_vs_actual.png
  - outputs/figures/residual_distribution.png
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

OUTPUTS_DIR = "outputs"
FIGURES_DIR = "outputs/figures"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def prepare_modeling_data():
    """
    Loads data and builds leakage-free candidate feature sets.
    """
    df = pd.read_csv("data/processed/feature_engineered_data.csv")
    # Filter out unassigned worker records for clean associative evaluation
    df = df[df["worker_id"] != "Unassigned"].copy()

    # Target
    y = df["picking_duration_minutes"].values

    # Pre-pick operational features (no target leakage)
    numeric_features = [
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
    categorical_features = [
        "item_category",
        "zone_type",
        "shift"
    ]

    encoded_cats = pd.get_dummies(df[categorical_features], drop_first=True, dtype=int)
    X = pd.concat([df[numeric_features], encoded_cats], axis=1)

    return X, y


def evaluate_models(X, y):
    """
    Trains and cross-validates multiple models on an 80/20 train/test split.
    """
    print(f"Dataset Size: {X.shape[0]} samples, {X.shape[1]} features")

    # 80/20 Train/Test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Baseline (Mean Predictor)": DummyRegressor(strategy="mean"),
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, min_samples_leaf=15, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42, n_jobs=-1)
    }

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    fitted_models = {}

    for name, model in models.items():
        print(f"Training & evaluating: {name}...")

        # Cross-validation R2 on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

        # Train on entire training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

        results.append({
            "Model": name,
            "CV_R2_Mean": round(cv_scores.mean(), 4),
            "CV_R2_Std": round(cv_scores.std(), 4),
            "Train_R2": round(train_r2, 4),
            "Test_R2": round(test_r2, 4),
            "Train_MAE_min": round(train_mae, 3),
            "Test_MAE_min": round(test_mae, 3),
            "Train_RMSE_min": round(train_rmse, 3),
            "Test_RMSE_min": round(test_rmse, 3),
            "Test_MAPE_pct": round(test_mape, 2)
        })

    results_df = pd.DataFrame(results).sort_values(by="Test_R2", ascending=False).reset_index(drop=True)
    report_path = os.path.join(OUTPUTS_DIR, "model_performance_report.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\nModel performance report exported to {report_path}")

    # Generate visual artifacts
    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    y_test_pred_best = best_model.predict(X_test)
    residuals = y_test - y_test_pred_best

    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    comp_df = results_df[results_df["Model"] != "Baseline (Mean Predictor)"].copy()
    bars = plt.barh(comp_df["Model"], comp_df["Test_R2"], color='#2b83ba', edgecolor='black', alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"R² = {w:.3f}", va='center', fontweight='bold', fontsize=9)
    plt.xlabel("Test Set Coefficient of Determination (R²)")
    plt.title("Predictive Model Comparison: Picking Duration (Minutes)", fontweight='bold', pad=12)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"))
    plt.close()

    # 2. Predicted vs Actual Scatter Plot (Best Model)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred_best, alpha=0.4, color='#1f77b4', edgecolors='none', s=35, label="Test Picks")
    min_val = min(y_test.min(), y_test_pred_best.min())
    max_val = max(y_test.max(), y_test_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect 1:1 Parity")
    plt.xlabel("Actual Picking Duration (Minutes)")
    plt.ylabel("Predicted Picking Duration (Minutes)")
    plt.title(f"Predicted vs. Actual Duration — {best_model_name}\nTest R² = {results_df.iloc[0]['Test_R2']:.3f}, Test MAE = {results_df.iloc[0]['Test_MAE_min']:.2f} min", fontweight='bold', pad=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "predicted_vs_actual.png"))
    plt.close()

    # 3. Residual Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=40, color='#e41a1c', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel("Prediction Error / Residual (Minutes: Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution ({best_model_name})\nMean Error = {residuals.mean():.3f} min, Std = {residuals.std():.3f} min", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residual_distribution.png"))
    plt.close()

    print("Generated model comparison, parity, and residual distribution charts.")
    return results_df, X_test, y_test, y_test_pred_best


if __name__ == "__main__":
    X, y = prepare_modeling_data()
    results_df, X_test, y_test, y_pred = evaluate_models(X, y)
    print("\n=== BENCHMARK MODEL PERFORMANCE SUMMARY ===")
    print(results_df.to_string(index=False))
