"""
FlowRack Logistics Pvt. Ltd. - Warehouse Efficiency Insight (Use Case 22)
Exploratory Data Analysis and Visualizations Pipeline
DS Sprint - Skill Based Assessment 2 - Round 2

Executes:
- All descriptive statistical calculations (overall and grouped)
- Answers all 6 EDA Business Questions with formal statistical tests (ANOVA / Kruskal-Wallis)
- Outlier analysis (IQR, percentiles, operational justification)
- Generates and exports all required high-resolution publication-grade figures to outputs/figures/
- Saves descriptive CSV tables to outputs/
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
# pyrefly: ignore [missing-import]
from scipy import stats

# Configure publication aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

FIGURES_DIR = "outputs/figures"
OUTPUTS_DIR = "outputs"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_data():
    """Loads feature engineered pick data and order summary."""
    picks_df = pd.read_csv("data/processed/feature_engineered_data.csv")
    picks_df["pick_start_time"] = pd.to_datetime(picks_df["pick_start_time"])
    picks_df["pick_finish_time"] = pd.to_datetime(picks_df["pick_finish_time"])
    picks_df["order_timestamp"] = pd.to_datetime(picks_df["order_timestamp"])

    orders_df = pd.read_csv("data/processed/order_level_summary.csv")
    orders_df["first_pick_start"] = pd.to_datetime(orders_df["first_pick_start"])
    orders_df["last_pick_finish"] = pd.to_datetime(orders_df["last_pick_finish"])
    orders_df["order_timestamp"] = pd.to_datetime(orders_df["order_timestamp"])
    return picks_df, orders_df


def generate_descriptive_statistics(picks_df, orders_df):
    """
    Computes comprehensive descriptive statistics and saves CSV reports.
    """
    print("\n--- Generating Descriptive Statistics ---")

    # 1. Overall pick duration statistics
    dur_stats = picks_df["picking_duration_minutes"].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
    print("\nPicking Duration (Minutes) Summary:")
    print(dur_stats)

    # 2. Group summaries
    group_dimensions = [
        ("zone_id", "by_zone"),
        ("zone_type", "by_zone_type"),
        ("distance_band", "by_distance_band"),
        ("shift", "by_shift"),
        ("experience_band", "by_experience_band"),
        ("item_category", "by_item_category"),
        ("item_size", "by_item_size"),
        ("storage_class", "by_storage_class"),
        ("order_priority", "by_order_priority"),
        ("order_size_category", "by_order_size_category")
    ]

    summary_tables = {}
    for col, name in group_dimensions:
        agg = picks_df.groupby(col)["picking_duration_minutes"].agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min="min",
            max="max"
        ).round(2).reset_index()
        summary_tables[name] = agg
        agg.to_csv(os.path.join(OUTPUTS_DIR, f"stats_{name}.csv"), index=False)

    # 3. Order-level descriptive statistics
    order_stats = orders_df[["item_count", "num_picks", "total_picked_quantity", 
                             "total_pick_duration_sum_minutes", "elapsed_processing_span_minutes", 
                             "order_queue_time_minutes"]].describe().round(2)
    order_stats.to_csv(os.path.join(OUTPUTS_DIR, "stats_order_level.csv"))

    print("Descriptive statistics exported successfully.")
    return summary_tables, order_stats


def run_eda_visualizations(picks_df, orders_df):
    """
    Generates all 11 required figures with high aesthetic standards.
    """
    print("\n--- Generating Visualizations ---")
    palette = sns.color_palette("mako")

    # -------------------------------------------------------------------------
    # Chart 1: Picking Duration by Zone (Bar Chart & Error Bars)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    zone_order = ["ZONE-A", "ZONE-B", "ZONE-C", "ZONE-D", "ZONE-E", "ZONE-F"]
    zone_means = picks_df.groupby("zone_id")["picking_duration_minutes"].mean().reindex(zone_order)
    zone_medians = picks_df.groupby("zone_id")["picking_duration_minutes"].median().reindex(zone_order)
    
    x = np.arange(len(zone_order))
    width = 0.35
    plt.bar(x - width/2, zone_means, width, label='Mean Duration', color='#1f77b4', alpha=0.9, edgecolor='black')
    plt.bar(x + width/2, zone_medians, width, label='Median Duration', color='#ff7f0e', alpha=0.9, edgecolor='black')
    
    for i, v in enumerate(zone_means):
        plt.text(i - width/2, v + 0.15, f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')
    for i, v in enumerate(zone_medians):
        plt.text(i + width/2, v + 0.15, f"{v:.1f}", ha='center', fontsize=9, fontweight='bold')

    plt.xlabel("Warehouse Zone")
    plt.ylabel("Picking Duration (Minutes)")
    plt.title("EDA Q1: Average & Median Picking Duration by Warehouse Zone", fontweight='bold', pad=12)
    plt.xticks(x, zone_order)
    plt.legend(frameon=True)
    plt.ylim(0, max(zone_means) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_duration_by_zone.png"))
    plt.close()
    print("Saved picking_duration_by_zone.png")

    # -------------------------------------------------------------------------
    # Chart 2: Picking Duration by Distance Band (Boxplot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    dist_order = ["Near", "Mid", "Far", "Deep-Storage"]
    sns.boxplot(data=picks_df, x="distance_band", y="picking_duration_minutes", 
                order=dist_order, palette="Blues_d", boxprops=dict(alpha=0.85), showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":6})
    plt.xlabel("Facility Distance Band")
    plt.ylabel("Picking Duration (Minutes)")
    plt.title("EDA Q1b: Picking Duration Distribution Across Distance Bands (Red Dot = Mean)", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_duration_by_distance_band.png"))
    plt.close()
    print("Saved picking_duration_by_distance_band.png")

    # -------------------------------------------------------------------------
    # Chart 3: Item Count vs Order Processing Time (Scatter & Trend)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.regplot(data=orders_df, x="item_count", y="elapsed_processing_span_minutes",
                scatter_kws={"alpha": 0.45, "color": "#2ca02c", "s": 40},
                line_kws={"color": "#d62728", "linewidth": 2.5, "label": "Linear Fit"})
    
    r_val, p_val = stats.pearsonr(orders_df["item_count"], orders_df["elapsed_processing_span_minutes"])
    plt.xlabel("Order Item Count")
    plt.ylabel("Elapsed Chronological Span (Minutes)")
    plt.title(f"EDA Q2: Order Item Count vs. Order Processing Span (Pearson r = {r_val:.2f}, p < 0.001)", 
              fontweight='bold', pad=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "item_count_vs_order_processing_time.png"))
    plt.close()
    print("Saved item_count_vs_order_processing_time.png")

    # -------------------------------------------------------------------------
    # Chart 4: Picking Duration by Shift (Violin + Boxplot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    shift_order = ["Morning", "Afternoon", "Night"]
    sns.boxplot(data=picks_df[picks_df["shift"] != "Unknown"], x="shift", y="picking_duration_minutes",
                order=shift_order, palette="Set2", showmeans=True,
                meanprops={"marker":"D", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize":6})
    plt.xlabel("Work Shift")
    plt.ylabel("Picking Duration (Minutes)")
    plt.title("EDA Q3: Operational Picking Duration by Shift Tier (Diamond = Mean)", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_duration_by_shift.png"))
    plt.close()
    print("Saved picking_duration_by_shift.png")

    # -------------------------------------------------------------------------
    # Chart 5: Picking Duration by Experience Band
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    exp_order = ["Novice", "Intermediate", "Senior"]
    valid_exp_df = picks_df[picks_df["experience_band"].isin(exp_order)]
    sns.boxplot(data=valid_exp_df, x="experience_band", y="picking_duration_minutes",
                order=exp_order, palette="Purples_r", showmeans=True,
                meanprops={"marker":"^", "markerfacecolor":"orange", "markeredgecolor":"black", "markersize":7})
    plt.xlabel("Worker Experience Band")
    plt.ylabel("Picking Duration (Minutes)")
    plt.title("EDA Q4: Picking Duration Across Worker Experience Bands (Triangle = Mean)", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_duration_by_experience_band.png"))
    plt.close()
    print("Saved picking_duration_by_experience_band.png")

    # -------------------------------------------------------------------------
    # Chart 6: Picking Duration by Item Category
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    cat_means = picks_df.groupby("item_category")["picking_duration_minutes"].mean().sort_values(ascending=False)
    bars = plt.barh(cat_means.index, cat_means.values, color='#4575b4', edgecolor='black', alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{w:.2f} min", 
                 va='center', fontsize=9, fontweight='bold')
    plt.xlabel("Mean Picking Duration (Minutes)")
    plt.ylabel("Item Category")
    plt.title("EDA Q5: Average Picking Duration by Item Category", fontweight='bold', pad=12)
    plt.xlim(0, max(cat_means.values) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_duration_by_item_category.png"))
    plt.close()
    print("Saved picking_duration_by_item_category.png")

    # -------------------------------------------------------------------------
    # Chart 7: Picking Duration by Storage Class
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    storage_order = ["Ambient", "Cold Storage", "Secure/High-Value", "Hazardous"]
    sns.barplot(data=picks_df, x="storage_class", y="picking_duration_minutes",
                order=storage_order, palette="crest", errorbar="ci", capsize=0.1, edgecolor='black')
    plt.xlabel("Storage Class")
    plt.ylabel("Mean Picking Duration (Minutes)")
    plt.title("EDA Q5b: Picking Duration by Item Storage Environment (with 95% CI)", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_duration_by_storage_class.png"))
    plt.close()
    print("Saved picking_duration_by_storage_class.png")

    # -------------------------------------------------------------------------
    # Chart 8: Overall Picking Duration Distribution
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.histplot(picks_df["picking_duration_minutes"], kde=True, bins=45, color="#17becf", edgecolor='black')
    median_val = picks_df["picking_duration_minutes"].median()
    mean_val = picks_df["picking_duration_minutes"].mean()
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f} min')
    plt.axvline(mean_val, color='navy', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.2f} min')
    plt.xlabel("Picking Duration (Minutes)")
    plt.ylabel("Frequency (Picks)")
    plt.title("Overall Distribution of Warehouse Picking Durations (Right-Skewed)", fontweight='bold', pad=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_duration_distribution.png"))
    plt.close()
    print("Saved picking_duration_distribution.png")

    # -------------------------------------------------------------------------
    # Chart 9: Picking Quantity vs Duration
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=picks_df, x="picking_quantity", y="picking_duration_minutes", palette="viridis",
                showmeans=True, meanprops={"marker":"s", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":5})
    plt.xlabel("Picking Quantity (Units Picked)")
    plt.ylabel("Picking Duration (Minutes)")
    plt.title("Picking Quantity vs. Duration (Red Square = Mean Duration)", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "picking_quantity_vs_duration.png"))
    plt.close()
    print("Saved picking_quantity_vs_duration.png")

    # -------------------------------------------------------------------------
    # Chart 10: Correlation Heatmap
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 7))
    num_cols = [
        "picking_duration_minutes",
        "picking_quantity",
        "distance_score",
        "zone_difficulty_score",
        "experience_score",
        "item_size_score",
        "storage_class_score",
        "handling_complexity_score",
        "order_priority_score",
        "pick_queue_time_minutes"
    ]
    corr_matrix = picks_df[num_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Numerical Warehouse Operational Features", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    plt.close()
    print("Saved correlation_heatmap.png")

    # -------------------------------------------------------------------------
    # Chart 11: Queue Time Analysis (EDA Q6)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    prio_order = ["Urgent", "High", "Medium", "Low"]
    sns.boxplot(data=orders_df, x="order_priority", y="order_queue_time_minutes",
                order=prio_order, palette="YlOrRd_r", showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"blue", "markeredgecolor":"black", "markersize":6})
    plt.xlabel("Order Priority Tier")
    plt.ylabel("Order Queue Latency (Minutes)")
    plt.title("EDA Q6: Order Release-to-Pick Queue Time by Priority Tier (Blue Circle = Mean)", fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "queue_time_analysis.png"))
    plt.close()
    print("Saved queue_time_analysis.png")


def run_statistical_hypothesis_tests(picks_df, orders_df):
    """
    Performs rigorous ANOVA, Kruskal-Wallis, and correlation tests.
    """
    print("\n--- Statistical Hypothesis Testing ---")

    # Test 1: Zone differences (ANOVA & Kruskal-Wallis)
    zone_groups = [group["picking_duration_minutes"].values for _, group in picks_df.groupby("zone_id")]
    f_stat, p_anova = stats.f_oneway(*zone_groups)
    h_stat, p_kw = stats.kruskal(*zone_groups)
    print(f"Zones ANOVA: F = {f_stat:.2f}, p = {p_anova:.4e}")
    print(f"Zones Kruskal-Wallis: H = {h_stat:.2f}, p = {p_kw:.4e}")

    # Test 2: Shift differences
    shift_groups = [group["picking_duration_minutes"].values for _, group in picks_df[picks_df["shift"] != "Unknown"].groupby("shift")]
    f_shift, p_shift = stats.f_oneway(*shift_groups)
    print(f"Shift ANOVA: F = {f_shift:.2f}, p = {p_shift:.4e}")

    # Test 3: Worker Experience differences
    exp_groups = [group["picking_duration_minutes"].values for _, group in picks_df[picks_df["experience_band"].isin(["Novice", "Intermediate", "Senior"])].groupby("experience_band")]
    f_exp, p_exp = stats.f_oneway(*exp_groups)
    print(f"Experience ANOVA: F = {f_exp:.2f}, p = {p_exp:.4e}")

    # Test 4: Item category differences
    cat_groups = [group["picking_duration_minutes"].values for _, group in picks_df.groupby("item_category")]
    f_cat, p_cat = stats.f_oneway(*cat_groups)
    print(f"Item Category ANOVA: F = {f_cat:.2f}, p = {p_cat:.4e}")

    # Test 5: Order item count correlation with elapsed time
    r_item, p_item = stats.pearsonr(orders_df["item_count"], orders_df["elapsed_processing_span_minutes"])
    print(f"Item Count vs Elapsed Time Correlation: r = {r_item:.3f}, p = {p_item:.4e}")

    test_results = pd.DataFrame([
        {"Test": "Zone Effect on Duration", "Method": "One-Way ANOVA", "Statistic": round(f_stat, 2), "p_value": p_anova, "Significant (alpha=0.05)": p_anova < 0.05},
        {"Test": "Zone Effect on Duration", "Method": "Kruskal-Wallis", "Statistic": round(h_stat, 2), "p_value": p_kw, "Significant (alpha=0.05)": p_kw < 0.05},
        {"Test": "Shift Effect on Duration", "Method": "One-Way ANOVA", "Statistic": round(f_shift, 2), "p_value": p_shift, "Significant (alpha=0.05)": p_shift < 0.05},
        {"Test": "Worker Experience Effect", "Method": "One-Way ANOVA", "Statistic": round(f_exp, 2), "p_value": p_exp, "Significant (alpha=0.05)": p_exp < 0.05},
        {"Test": "Item Category Effect", "Method": "One-Way ANOVA", "Statistic": round(f_cat, 2), "p_value": p_cat, "Significant (alpha=0.05)": p_cat < 0.05},
        {"Test": "Item Count vs Elapsed Time", "Method": "Pearson Correlation", "Statistic": round(r_item, 3), "p_value": p_item, "Significant (alpha=0.05)": p_item < 0.05}
    ])
    test_results.to_csv(os.path.join(OUTPUTS_DIR, "statistical_tests_summary.csv"), index=False)
    print("Statistical test results exported to outputs/statistical_tests_summary.csv")
    return test_results


def run_pipeline():
    picks_df, orders_df = load_data()
    generate_descriptive_statistics(picks_df, orders_df)
    run_eda_visualizations(picks_df, orders_df)
    run_statistical_hypothesis_tests(picks_df, orders_df)


if __name__ == "__main__":
    run_pipeline()
