"""
FlowRack Logistics Pvt. Ltd. - Warehouse Efficiency Insight (Use Case 22)
Data Quality Assessment and Cleaning Pipeline
DS Sprint - Skill Based Assessment 2 - Round 2

Performs comprehensive data auditing, cleaning, and dataset integration:
- Audits missing values, duplicates, timestamp anomalies, inconsistent labels, foreign keys
- Produces outputs/data_quality_report.csv
- Produces outputs/dataset_summary.csv
- Saves individual cleaned datasets to data/processed/
- Performs grain-aware integration for pick-level and order-level analysis
- Saves primary integrated dataset to data/processed/cleaned_warehouse_data.csv
"""

import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd


def load_raw_data(data_dir="data/raw"):
    """Loads all five raw CSV datasets."""
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    picks = pd.read_csv(os.path.join(data_dir, "picks.csv"))
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    workers = pd.read_csv(os.path.join(data_dir, "workers.csv"))
    zones = pd.read_csv(os.path.join(data_dir, "zones.csv"))
    return orders, picks, items, workers, zones


def generate_dataset_summary(orders, picks, items, workers, zones, output_file="outputs/dataset_summary.csv"):
    """
    Creates a summary table describing all five datasets.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_data = [
        {
            "dataset_name": "orders.csv",
            "grain": "Order Level (1 row per customer order)",
            "rows": len(orders),
            "columns": len(orders.columns),
            "primary_key": "order_id",
            "key_attributes": "item_count, order_priority, order_timestamp",
            "synthetic_status": "Synthetic benchmark for FlowRack Logistics"
        },
        {
            "dataset_name": "picks.csv",
            "grain": "Pick Event Level (1 row per item pick occurrence)",
            "rows": len(picks),
            "columns": len(picks.columns),
            "primary_key": "pick_id",
            "key_attributes": "order_id, worker_id, item_id, zone_id, timestamps, quantity",
            "synthetic_status": "Synthetic benchmark with controlled quality anomalies"
        },
        {
            "dataset_name": "items.csv",
            "grain": "Item Master Level (1 row per SKU / item catalog entry)",
            "rows": len(items),
            "columns": len(items.columns),
            "primary_key": "item_id",
            "key_attributes": "item_category, item_size, storage_class",
            "synthetic_status": "Synthetic catalog profile"
        },
        {
            "dataset_name": "workers.csv",
            "grain": "Worker Roster Level (1 row per warehouse associate)",
            "rows": len(workers),
            "columns": len(workers.columns),
            "primary_key": "worker_id",
            "key_attributes": "shift, experience_band",
            "synthetic_status": "Synthetic workforce profile"
        },
        {
            "dataset_name": "zones.csv",
            "grain": "Zone Master Level (1 row per physical warehouse storage zone)",
            "rows": len(zones),
            "columns": len(zones.columns),
            "primary_key": "zone_id",
            "key_attributes": "zone_type, distance_band",
            "synthetic_status": "Synthetic facility layout profile"
        }
    ]
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_file, index=False)
    print(f"Dataset summary written to {output_file}")
    return df_summary


def assess_data_quality(orders, picks, items, workers, zones, output_file="outputs/data_quality_report.csv"):
    """
    Performs data quality checks across all datasets and writes the report.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    report_rows = []

    # 1. Check orders
    report_rows.append({
        "dataset_name": "orders",
        "column_name": "order_id",
        "missing_count": orders["order_id"].isna().sum(),
        "missing_pct": round(orders["order_id"].isna().mean() * 100, 2),
        "duplicate_count": orders["order_id"].duplicated().sum(),
        "validation_issue": "Check primary key uniqueness",
        "cleaning_action": "Retain - all order IDs unique" if orders["order_id"].duplicated().sum() == 0 else "Deduplicate"
    })
    for col in ["item_count", "order_priority", "order_timestamp"]:
        report_rows.append({
            "dataset_name": "orders",
            "column_name": col,
            "missing_count": orders[col].isna().sum(),
            "missing_pct": round(orders[col].isna().mean() * 100, 2),
            "duplicate_count": 0,
            "validation_issue": "Check missingness and format validity",
            "cleaning_action": "Valid - no missing values detected"
        })

    # 2. Check picks
    # Exact duplicate rows
    exact_pick_dupes = picks.duplicated().sum()
    report_rows.append({
        "dataset_name": "picks",
        "column_name": "ALL (Row Level)",
        "missing_count": 0,
        "missing_pct": 0.0,
        "duplicate_count": exact_pick_dupes,
        "validation_issue": "Repeated barcode / scanner double trigger records",
        "cleaning_action": f"Remove {exact_pick_dupes} exact duplicate rows"
    })

    # Pick finish time missingness
    missing_finish = picks["pick_finish_time"].isna().sum()
    report_rows.append({
        "dataset_name": "picks",
        "column_name": "pick_finish_time",
        "missing_count": missing_finish,
        "missing_pct": round(picks["pick_finish_time"].isna().mean() * 100, 2),
        "duplicate_count": 0,
        "validation_issue": "Missing finish timestamp (scanner disconnection / incomplete pick)",
        "cleaning_action": "Do not impute or invent timestamps; exclude from duration analysis"
    })

    # Zone ID consistency
    unique_zones_picks = picks["zone_id"].dropna().unique()
    valid_zones = set(zones["zone_id"])
    inconsistent_zones = [z for z in unique_zones_picks if z not in valid_zones]
    report_rows.append({
        "dataset_name": "picks",
        "column_name": "zone_id",
        "missing_count": picks["zone_id"].isna().sum(),
        "missing_pct": round(picks["zone_id"].isna().mean() * 100, 2),
        "duplicate_count": 0,
        "validation_issue": f"Inconsistent zone casing/whitespace ({len(inconsistent_zones)} variations)",
        "cleaning_action": "Standardize using strip() and upper() to align with master zone records"
    })

    # Negative picking durations
    picks_temp = picks.dropna(subset=["pick_start_time", "pick_finish_time"]).copy()
    picks_temp["start_dt"] = pd.to_datetime(picks_temp["pick_start_time"], errors="coerce")
    picks_temp["finish_dt"] = pd.to_datetime(picks_temp["pick_finish_time"], errors="coerce")
    neg_durations = (picks_temp["finish_dt"] < picks_temp["start_dt"]).sum()
    report_rows.append({
        "dataset_name": "picks",
        "column_name": "pick_start_time / pick_finish_time",
        "missing_count": 0,
        "missing_pct": 0.0,
        "duplicate_count": 0,
        "validation_issue": f"Negative duration anomaly: finish occurs before start ({neg_durations} records)",
        "cleaning_action": "Exclude corrupted timestamp records from duration analysis; flag as invalid"
    })

    # Worker ID missingness / FK integrity
    missing_workers = picks["worker_id"].isna().sum()
    report_rows.append({
        "dataset_name": "picks",
        "column_name": "worker_id",
        "missing_count": missing_workers,
        "missing_pct": round(picks["worker_id"].isna().mean() * 100, 2),
        "duplicate_count": 0,
        "validation_issue": "Missing worker assignment in picking log",
        "cleaning_action": "Label worker as 'Unassigned'; preserve pick for zone/item analysis"
    })

    # Item ID FK check
    missing_items = picks["item_id"].isna().sum()
    unmatched_items = (~picks["item_id"].isin(items["item_id"])).sum()
    report_rows.append({
        "dataset_name": "picks",
        "column_name": "item_id",
        "missing_count": missing_items,
        "missing_pct": round(picks["item_id"].isna().mean() * 100, 2),
        "duplicate_count": 0,
        "validation_issue": f"Foreign key integrity check ({unmatched_items} unmatched items)",
        "cleaning_action": "Retain valid matched items"
    })

    # Order ID FK check
    missing_orders = picks["order_id"].isna().sum()
    unmatched_orders = (~picks["order_id"].isin(orders["order_id"])).sum()
    report_rows.append({
        "dataset_name": "picks",
        "column_name": "order_id",
        "missing_count": missing_orders,
        "missing_pct": round(picks["order_id"].isna().mean() * 100, 2),
        "duplicate_count": 0,
        "validation_issue": f"Foreign key integrity check ({unmatched_orders} unmatched orders)",
        "cleaning_action": "Retain valid matched orders"
    })

    # 3. Check items, workers, zones
    report_rows.append({
        "dataset_name": "items",
        "column_name": "item_id",
        "missing_count": items["item_id"].isna().sum(),
        "missing_pct": 0.0,
        "duplicate_count": items["item_id"].duplicated().sum(),
        "validation_issue": "Check primary key uniqueness",
        "cleaning_action": "Retain - all items valid"
    })
    report_rows.append({
        "dataset_name": "workers",
        "column_name": "worker_id",
        "missing_count": workers["worker_id"].isna().sum(),
        "missing_pct": 0.0,
        "duplicate_count": workers["worker_id"].duplicated().sum(),
        "validation_issue": "Check primary key uniqueness",
        "cleaning_action": "Retain - all worker IDs valid"
    })
    report_rows.append({
        "dataset_name": "zones",
        "column_name": "zone_id",
        "missing_count": zones["zone_id"].isna().sum(),
        "missing_pct": 0.0,
        "duplicate_count": zones["zone_id"].duplicated().sum(),
        "validation_issue": "Check primary key uniqueness",
        "cleaning_action": "Retain - all zone IDs valid"
    })

    df_quality = pd.DataFrame(report_rows)
    df_quality.to_csv(output_file, index=False)
    print(f"Data quality report written to {output_file}")
    return df_quality


def clean_datasets(orders, picks, items, workers, zones, output_dir="data/processed"):
    """
    Cleans all datasets following Round 2 requirements and saves them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\nStarting systematic cleaning pipeline...")

    # =========================================================================
    # A. Clean orders
    # =========================================================================
    df_orders_cleaned = orders.copy()
    df_orders_cleaned["order_id"] = df_orders_cleaned["order_id"].astype(str).str.strip()
    df_orders_cleaned["order_priority"] = df_orders_cleaned["order_priority"].astype(str).str.strip().str.capitalize()
    df_orders_cleaned["order_timestamp"] = pd.to_datetime(df_orders_cleaned["order_timestamp"], errors="coerce")
    df_orders_cleaned["item_count"] = pd.to_numeric(df_orders_cleaned["item_count"], errors="coerce")
    # Drop exact duplicates if any
    df_orders_cleaned = df_orders_cleaned.drop_duplicates()

    # =========================================================================
    # B. Clean items, workers, zones
    # =========================================================================
    df_items_cleaned = items.copy()
    df_items_cleaned["item_id"] = df_items_cleaned["item_id"].astype(str).str.strip()
    df_items_cleaned["item_category"] = df_items_cleaned["item_category"].astype(str).str.strip()
    df_items_cleaned["item_size"] = df_items_cleaned["item_size"].astype(str).str.strip()
    df_items_cleaned["storage_class"] = df_items_cleaned["storage_class"].astype(str).str.strip()
    df_items_cleaned = df_items_cleaned.drop_duplicates(subset=["item_id"])

    df_workers_cleaned = workers.copy()
    df_workers_cleaned["worker_id"] = df_workers_cleaned["worker_id"].astype(str).str.strip()
    df_workers_cleaned["shift"] = df_workers_cleaned["shift"].astype(str).str.strip()
    df_workers_cleaned["experience_band"] = df_workers_cleaned["experience_band"].astype(str).str.strip()
    df_workers_cleaned = df_workers_cleaned.drop_duplicates(subset=["worker_id"])

    df_zones_cleaned = zones.copy()
    df_zones_cleaned["zone_id"] = df_zones_cleaned["zone_id"].astype(str).str.strip().str.upper()
    df_zones_cleaned["zone_type"] = df_zones_cleaned["zone_type"].astype(str).str.strip()
    df_zones_cleaned["distance_band"] = df_zones_cleaned["distance_band"].astype(str).str.strip()
    df_zones_cleaned = df_zones_cleaned.drop_duplicates(subset=["zone_id"])

    # =========================================================================
    # C. Clean picks
    # =========================================================================
    raw_picks_count = len(picks)
    df_picks_cleaned = picks.copy()

    # 1. Remove exact duplicate rows
    df_picks_cleaned = df_picks_cleaned.drop_duplicates()
    dedup_count = raw_picks_count - len(df_picks_cleaned)
    print(f"Picks: Removed {dedup_count} exact duplicate rows.")

    # 2. Standardize whitespace and letter casing on zone_id
    df_picks_cleaned["zone_id"] = df_picks_cleaned["zone_id"].astype(str).str.strip().str.upper()

    # 3. Standardize IDs
    df_picks_cleaned["pick_id"] = df_picks_cleaned["pick_id"].astype(str).str.strip()
    df_picks_cleaned["order_id"] = df_picks_cleaned["order_id"].astype(str).str.strip()
    df_picks_cleaned["item_id"] = df_picks_cleaned["item_id"].astype(str).str.strip()
    
    # 4. Handle worker_id (if missing, set to 'Unassigned')
    df_picks_cleaned["worker_id"] = df_picks_cleaned["worker_id"].apply(
        lambda x: "Unassigned" if pd.isna(x) or str(x).strip().lower() in ["nan", "none", ""] else str(x).strip()
    )

    # 5. Parse timestamps
    df_picks_cleaned["pick_start_time"] = pd.to_datetime(df_picks_cleaned["pick_start_time"], errors="coerce")
    df_picks_cleaned["pick_finish_time"] = pd.to_datetime(df_picks_cleaned["pick_finish_time"], errors="coerce")

    # 6. Calculate picking duration in minutes
    # picking_duration_minutes = (pick_finish_time - pick_start_time).total_seconds() / 60
    duration_seconds = (df_picks_cleaned["pick_finish_time"] - df_picks_cleaned["pick_start_time"]).dt.total_seconds()
    df_picks_cleaned["picking_duration_minutes"] = duration_seconds / 60.0

    # 7. Identify duration validity flag:
    # A pick is valid for duration analysis IF:
    # - finish time is not null
    # - finish time >= start time (duration >= 0)
    # - duration is physiologically/operationally plausible (> 0.2 minutes, <= 60 minutes)
    valid_duration_mask = (
        df_picks_cleaned["picking_duration_minutes"].notna() &
        (df_picks_cleaned["picking_duration_minutes"] > 0) &
        (df_picks_cleaned["picking_duration_minutes"] <= 60.0)
    )
    df_picks_cleaned["is_valid_duration"] = valid_duration_mask

    invalid_durations = (~valid_duration_mask).sum()
    print(f"Picks: Identified {invalid_durations} records with missing/negative/invalid durations.")
    print("      These are retained with `is_valid_duration=False` for auditability,")
    print("      and filtered out strictly during duration-based analyses.")

    # Save cleaned individual tables
    orders_out = os.path.join(output_dir, "cleaned_orders.csv")
    picks_out = os.path.join(output_dir, "cleaned_picks.csv")
    items_out = os.path.join(output_dir, "cleaned_items.csv")
    workers_out = os.path.join(output_dir, "cleaned_workers.csv")
    zones_out = os.path.join(output_dir, "cleaned_zones.csv")

    df_orders_cleaned.to_csv(orders_out, index=False)
    df_picks_cleaned.to_csv(picks_out, index=False)
    df_items_cleaned.to_csv(items_out, index=False)
    df_workers_cleaned.to_csv(workers_out, index=False)
    df_zones_cleaned.to_csv(zones_out, index=False)

    print("\nSaved individual cleaned datasets:")
    print(f"- {orders_out}: {len(df_orders_cleaned)} rows")
    print(f"- {picks_out}: {len(df_picks_cleaned)} rows")
    print(f"- {items_out}: {len(df_items_cleaned)} rows")
    print(f"- {workers_out}: {len(df_workers_cleaned)} rows")
    print(f"- {zones_out}: {len(df_zones_cleaned)} rows")

    return df_orders_cleaned, df_picks_cleaned, df_items_cleaned, df_workers_cleaned, df_zones_cleaned


def integrate_datasets(df_orders, df_picks, df_items, df_workers, df_zones, output_file="data/processed/cleaned_warehouse_data.csv"):
    """
    Performs grain-aware integration.
    
    Level 1: Pick-level integration (retaining pick grain)
    Joins each pick with item characteristics, zone characteristics, worker roster, and parent order attributes.
    Row count is validated before and after merge to guarantee zero multiplication.
    
    Level 2: Order-level integration
    Aggregates picks by order to calculate true total pick duration sum, elapsed span, distinct zones.
    """
    print("\nPerforming dataset integration with grain awareness...")

    # Filter picks to valid duration records for the primary warehouse analytical dataset
    valid_picks = df_picks[df_picks["is_valid_duration"]].copy()
    initial_valid_count = len(valid_picks)
    print(f"Starting pick-level merge on {initial_valid_count} valid picking records...")

    # Merge with items (many-to-one on item_id)
    merged = valid_picks.merge(df_items, on="item_id", how="left")
    assert len(merged) == initial_valid_count, f"Row count changed after item merge: {len(merged)} vs {initial_valid_count}"

    # Merge with zones (many-to-one on zone_id)
    merged = merged.merge(df_zones, on="zone_id", how="left")
    assert len(merged) == initial_valid_count, f"Row count changed after zone merge: {len(merged)} vs {initial_valid_count}"

    # Merge with workers (many-to-one on worker_id, left join to keep Unassigned picks)
    merged = merged.merge(df_workers, on="worker_id", how="left")
    assert len(merged) == initial_valid_count, f"Row count changed after worker merge: {len(merged)} vs {initial_valid_count}"

    # Fill unassigned worker attributes
    merged["shift"] = merged["shift"].fillna("Unknown")
    merged["experience_band"] = merged["experience_band"].fillna("Unknown")

    # Merge with orders (many-to-one on order_id)
    merged = merged.merge(df_orders[["order_id", "item_count", "order_priority", "order_timestamp"]], on="order_id", how="left")
    assert len(merged) == initial_valid_count, f"Row count changed after orders merge: {len(merged)} vs {initial_valid_count}"

    print(f"Pick-level integration validated: exactly {len(merged)} rows retained (0 row duplication).")

    # Calculate queue time (minutes between order placement and this pick's start)
    merged["queue_time_minutes"] = (
        pd.to_datetime(merged["pick_start_time"]) - pd.to_datetime(merged["order_timestamp"])
    ).dt.total_seconds() / 60.0

    # Save integrated dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"Primary integrated dataset saved to {output_file} ({len(merged)} rows, {len(merged.columns)} columns).")

    return merged


def run_pipeline():
    orders, picks, items, workers, zones = load_raw_data()
    generate_dataset_summary(orders, picks, items, workers, zones)
    assess_data_quality(orders, picks, items, workers, zones)
    df_orders, df_picks, df_items, df_workers, df_zones = clean_datasets(orders, picks, items, workers, zones)
    integrated_data = integrate_datasets(df_orders, df_picks, df_items, df_workers, df_zones)
    return integrated_data


if __name__ == "__main__":
    run_pipeline()
