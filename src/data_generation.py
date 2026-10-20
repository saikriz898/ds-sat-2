"""
FlowRack Logistics Pvt. Ltd. - Warehouse Efficiency Insight (Use Case 22)
Synthetic Data Generation Script
DS Sprint - Skill Based Assessment 2 - Round 2

Generates realistic synthetic datasets for warehouse picking and order processing:
- orders.csv (1,000 orders)
- items.csv (120 items)
- workers.csv (40 workers)
- zones.csv (6 zones)
- picks.csv (~4,700 picking events with controlled data quality anomalies)

All data is generated with a fixed random seed (42) for exact reproducibility.
"""

import os
# pyrefly: ignore [missing-import]
import random
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set fixed random seeds for exact reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_synthetic_data(output_dir="data/raw"):
    """
    Generates all five raw synthetic datasets and saves them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Generating synthetic datasets for FlowRack Logistics Pvt. Ltd. (Seed: 42)...")

    # =========================================================================
    # 1. ZONES DATASET (zones.csv)
    # =========================================================================
    zones_data = [
        {"zone_id": "ZONE-A", "zone_type": "Fast-Moving", "distance_band": "Near"},
        {"zone_id": "ZONE-B", "zone_type": "Standard Shelf", "distance_band": "Near"},
        {"zone_id": "ZONE-C", "zone_type": "Mezzanine", "distance_band": "Mid"},
        {"zone_id": "ZONE-D", "zone_type": "High-Rack", "distance_band": "Mid"},
        {"zone_id": "ZONE-E", "zone_type": "Cold Vault", "distance_band": "Far"},
        {"zone_id": "ZONE-F", "zone_type": "Bulk Storage", "distance_band": "Deep-Storage"}
    ]
    df_zones = pd.DataFrame(zones_data)
    zones_file = os.path.join(output_dir, "zones.csv")
    df_zones.to_csv(zones_file, index=False)
    print(f"Saved {zones_file}: {len(df_zones)} zones.")

    # =========================================================================
    # 2. ITEMS DATASET (items.csv)
    # =========================================================================
    categories = ["Electronics", "Apparel", "Groceries", "Heavy Equipment", "Home Goods", "Pharmaceuticals"]
    cat_weights = [0.20, 0.25, 0.20, 0.10, 0.15, 0.10]
    
    items_list = []
    num_items = 120
    for i in range(1, num_items + 1):
        item_id = f"ITM-{i:03d}"
        category = random.choices(categories, weights=cat_weights, k=1)[0]
        
        # Realistic size and storage class logic based on category
        if category == "Heavy Equipment":
            size = random.choices(["Large", "Bulky"], weights=[0.3, 0.7], k=1)[0]
            storage = random.choices(["Ambient", "Hazardous"], weights=[0.8, 0.2], k=1)[0]
        elif category == "Pharmaceuticals":
            size = random.choices(["Small", "Medium"], weights=[0.8, 0.2], k=1)[0]
            storage = random.choices(["Cold Storage", "Secure/High-Value"], weights=[0.6, 0.4], k=1)[0]
        elif category == "Electronics":
            size = random.choices(["Small", "Medium", "Large"], weights=[0.5, 0.4, 0.1], k=1)[0]
            storage = random.choices(["Secure/High-Value", "Ambient"], weights=[0.65, 0.35], k=1)[0]
        elif category == "Groceries":
            size = random.choices(["Small", "Medium", "Large"], weights=[0.4, 0.4, 0.2], k=1)[0]
            storage = random.choices(["Ambient", "Cold Storage"], weights=[0.6, 0.4], k=1)[0]
        elif category == "Apparel":
            size = random.choices(["Small", "Medium"], weights=[0.6, 0.4], k=1)[0]
            storage = "Ambient"
        else:  # Home Goods
            size = random.choices(["Medium", "Large", "Bulky"], weights=[0.5, 0.35, 0.15], k=1)[0]
            storage = "Ambient"

        items_list.append({
            "item_id": item_id,
            "item_category": category,
            "item_size": size,
            "storage_class": storage
        })
    df_items = pd.DataFrame(items_list)
    items_file = os.path.join(output_dir, "items.csv")
    df_items.to_csv(items_file, index=False)
    print(f"Saved {items_file}: {len(df_items)} items.")

    # Map items to primary zones for realistic operational assignment
    def get_preferred_zone(storage, size):
        if storage == "Cold Storage":
            return "ZONE-E"
        elif size == "Bulky" or storage == "Hazardous":
            return "ZONE-F"
        elif storage == "Secure/High-Value":
            return "ZONE-D"
        elif size == "Large":
            return "ZONE-D"
        elif size == "Small":
            return random.choice(["ZONE-A", "ZONE-B"])
        else:
            return random.choice(["ZONE-B", "ZONE-C"])

    item_preferred_zone = {
        row["item_id"]: get_preferred_zone(row["storage_class"], row["item_size"])
        for _, row in df_items.iterrows()
    }

    # =========================================================================
    # 3. WORKERS DATASET (workers.csv)
    # =========================================================================
    num_workers = 40
    shifts = ["Morning", "Afternoon", "Night"]
    shift_weights = [0.45, 0.35, 0.20]
    exp_bands = ["Novice", "Intermediate", "Senior"]
    exp_weights = [0.30, 0.45, 0.25]

    workers_list = []
    for i in range(1, num_workers + 1):
        worker_id = f"WRK-{i:03d}"
        shift = random.choices(shifts, weights=shift_weights, k=1)[0]
        exp = random.choices(exp_bands, weights=exp_weights, k=1)[0]
        workers_list.append({
            "worker_id": worker_id,
            "shift": shift,
            "experience_band": exp
        })
    df_workers = pd.DataFrame(workers_list)
    workers_file = os.path.join(output_dir, "workers.csv")
    df_workers.to_csv(workers_file, index=False)
    print(f"Saved {workers_file}: {len(df_workers)} workers.")

    # =========================================================================
    # 4. ORDERS DATASET (orders.csv)
    # =========================================================================
    num_orders = 1000
    base_date = datetime(2026, 3, 1, 6, 0, 0)
    
    priorities = ["Low", "Medium", "High", "Urgent"]
    prio_weights = [0.20, 0.50, 0.20, 0.10]
    
    orders_list = []
    order_timestamps = []
    current_time = base_date

    for i in range(1, num_orders + 1):
        order_id = f"ORD-{i:04d}"
        # Inter-arrival times (average 15-20 minutes, simulating continuous warehouse releases over 14 days)
        arrival_gap_minutes = np.random.exponential(scale=18.0)
        current_time += timedelta(minutes=float(arrival_gap_minutes))
        
        # Item count distribution: skewed, typical 1-12 items, right tail
        # 1-3 (Small, ~40%), 4-7 (Medium, ~40%), 8-12 (Large, ~15%), 13-16 (Bulk, ~5%)
        rand_val = random.random()
        if rand_val < 0.40:
            item_count = random.randint(1, 3)
        elif rand_val < 0.80:
            item_count = random.randint(4, 7)
        elif rand_val < 0.95:
            item_count = random.randint(8, 12)
        else:
            item_count = random.randint(13, 16)

        priority = random.choices(priorities, weights=prio_weights, k=1)[0]
        
        orders_list.append({
            "order_id": order_id,
            "item_count": item_count,
            "order_priority": priority,
            "order_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
        })
        order_timestamps.append(current_time)

    df_orders = pd.DataFrame(orders_list)
    orders_file = os.path.join(output_dir, "orders.csv")
    df_orders.to_csv(orders_file, index=False)
    print(f"Saved {orders_file}: {len(df_orders)} orders.")

    # =========================================================================
    # 5. PICKS DATASET (picks.csv)
    # =========================================================================
    picks_list = []
    pick_counter = 1

    # Worker lookup
    worker_shifts = dict(zip(df_workers["worker_id"], df_workers["shift"]))
    worker_exp = dict(zip(df_workers["worker_id"], df_workers["experience_band"]))
    shift_workers = {
        "Morning": df_workers[df_workers["shift"] == "Morning"]["worker_id"].tolist(),
        "Afternoon": df_workers[df_workers["shift"] == "Afternoon"]["worker_id"].tolist(),
        "Night": df_workers[df_workers["shift"] == "Night"]["worker_id"].tolist()
    }

    # Zone characteristics lookup
    zone_dist_lookup = dict(zip(df_zones["zone_id"], df_zones["distance_band"]))
    zone_type_lookup = dict(zip(df_zones["zone_id"], df_zones["zone_type"]))
    item_lookup = df_items.set_index("item_id").to_dict(orient="index")

    for idx, ord_row in df_orders.iterrows():
        order_id = ord_row["order_id"]
        item_count = ord_row["item_count"]
        priority = ord_row["order_priority"]
        order_dt = order_timestamps[idx]

        # Queue time before first pick starts based on order priority
        # Urgent: 5 - 20 mins; High: 15 - 40 mins; Medium: 25 - 60 mins; Low: 40 - 90 mins
        if priority == "Urgent":
            queue_minutes = np.random.uniform(5, 20)
        elif priority == "High":
            queue_minutes = np.random.uniform(15, 40)
        elif priority == "Medium":
            queue_minutes = np.random.uniform(25, 60)
        else:
            queue_minutes = np.random.uniform(40, 95)
        
        first_pick_start = order_dt + timedelta(minutes=float(queue_minutes))
        current_pick_time = first_pick_start

        # Determine shift at time of picking
        hour = current_pick_time.hour
        if 6 <= hour < 14:
            current_shift = "Morning"
        elif 14 <= hour < 22:
            current_shift = "Afternoon"
        else:
            current_shift = "Night"

        # Select items for this order
        selected_item_ids = random.sample(df_items["item_id"].tolist(), k=item_count)

        for itm_id in selected_item_ids:
            pick_id = f"PCK-{pick_counter:05d}"
            pick_counter += 1

            # Worker on current shift
            assigned_worker = random.choice(shift_workers[current_shift])
            worker_experience = worker_exp[assigned_worker]

            # Zone assignment (mostly preferred zone for the item)
            zone_id = item_preferred_zone[itm_id]
            # Occasional overflow (5%)
            if random.random() < 0.05:
                zone_id = random.choice(df_zones["zone_id"].tolist())

            distance_band = zone_dist_lookup[zone_id]
            zone_type = zone_type_lookup[zone_id]
            itm_info = item_lookup[itm_id]
            itm_size = itm_info["item_size"]
            storage = itm_info["storage_class"]

            # Picking quantity: 1-10 units
            # Skewed towards 1-3 units
            picking_qty = int(np.random.choice([1, 2, 3, 4, 5, 8, 10], p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]))

            # Realistic picking duration formula (minutes)
            # Baseline: 3.5 mins
            duration = 3.5

            # Distance band contribution
            dist_add = {"Near": 0.0, "Mid": 1.4, "Far": 3.1, "Deep-Storage": 4.6}
            duration += dist_add.get(distance_band, 0.0)

            # Zone type contribution
            type_add = {
                "Fast-Moving": -0.6,
                "Standard Shelf": 0.2,
                "Mezzanine": 1.2,
                "High-Rack": 2.2,     # Forklift handling
                "Cold Vault": 1.8,    # PPE & thermal access
                "Bulk Storage": 2.6   # Heavy pallet operations
            }
            duration += type_add.get(zone_type, 0.0)

            # Item size contribution
            size_add = {"Small": 0.0, "Medium": 0.8, "Large": 2.0, "Bulky": 4.2}
            duration += size_add.get(itm_size, 0.0)

            # Storage class contribution
            storage_add = {"Ambient": 0.0, "Cold Storage": 1.1, "Secure/High-Value": 1.5, "Hazardous": 2.3}
            duration += storage_add.get(storage, 0.0)

            # Worker experience contribution (Senior faster, Novice slower)
            exp_mod = {"Senior": -1.3, "Intermediate": -0.3, "Novice": 1.2}
            duration += exp_mod.get(worker_experience, 0.0)

            # Shift contribution (Night shift slightly slower / fatigue)
            shift_mod = {"Morning": 0.0, "Afternoon": 0.2, "Night": 0.7}
            duration += shift_mod.get(current_shift, 0.0)

            # Quantity contribution (+0.32 min per item beyond 1)
            duration += 0.32 * (picking_qty - 1)

            # Random natural variability (log-normal residual variation)
            noise = np.random.lognormal(mean=0.0, sigma=0.22)
            duration = duration * noise
            # Minimum plausible duration: 1.0 min
            duration = max(1.0, float(duration))

            # Pick start time
            pick_start = current_pick_time
            pick_finish = pick_start + timedelta(minutes=duration)

            # Move current pick time forward slightly for sequential picking in the order
            current_pick_time = pick_finish + timedelta(minutes=float(np.random.uniform(0.5, 3.0)))

            picks_list.append({
                "pick_id": pick_id,
                "order_id": order_id,
                "worker_id": assigned_worker,
                "item_id": itm_id,
                "zone_id": zone_id,
                "pick_start_time": pick_start.strftime("%Y-%m-%d %H:%M:%S"),
                "pick_finish_time": pick_finish.strftime("%Y-%m-%d %H:%M:%S"),
                "picking_quantity": picking_qty
            })

    df_picks = pd.DataFrame(picks_list)
    print(f"Base picks generated: {len(df_picks)} records.")

    # =========================================================================
    # INJECT CONTROLLED DATA QUALITY ANOMALIES FOR ROUND 2 DEMONSTRATION
    # =========================================================================
    print("Injecting controlled data quality anomalies...")

    # Issue 1: Missing finish times (~25 records, e.g. RF scanner crash / system timeout)
    missing_finish_indices = np.random.choice(df_picks.index, size=25, replace=False)
    df_picks.loc[missing_finish_indices, "pick_finish_time"] = np.nan

    # Issue 2: Inconsistent zone labels (~35 records with whitespace or letter case variations)
    # Available indices not in missing_finish
    available_indices = list(set(df_picks.index) - set(missing_finish_indices))
    inconsistent_zone_indices = np.random.choice(available_indices, size=35, replace=False)
    for idx in inconsistent_zone_indices:
        current_zone = df_picks.loc[idx, "zone_id"]
        variation = random.choice([
            current_zone.lower(),               # 'zone-a'
            f" {current_zone} ",                # ' ZONE-A '
            f"{current_zone.lower()} ",         # 'zone-b '
            current_zone.capitalize()           # 'Zone-a'
        ])
        df_picks.loc[idx, "zone_id"] = variation

    # Issue 3: Negative picking durations caused by invalid timestamps (5 records)
    neg_duration_indices = np.random.choice(available_indices, size=5, replace=False)
    for idx in neg_duration_indices:
        # Swap start and finish or set finish earlier
        start_ts = pd.to_datetime(df_picks.loc[idx, "pick_start_time"])
        bogus_finish = start_ts - timedelta(minutes=float(np.random.uniform(5, 25)))
        df_picks.loc[idx, "pick_finish_time"] = bogus_finish.strftime("%Y-%m-%d %H:%M:%S")

    # Issue 4: Missing worker ID (~12 records, unassigned or detached scanner)
    missing_worker_indices = np.random.choice(available_indices, size=12, replace=False)
    df_picks.loc[missing_worker_indices, "worker_id"] = np.nan

    # Issue 5: Duplicate picking records (~18 duplicate rows simulating repeated barcode scanner scans)
    duplicate_indices = np.random.choice(available_indices, size=18, replace=False)
    duplicate_rows = df_picks.loc[duplicate_indices].copy()
    df_picks = pd.concat([df_picks, duplicate_rows], ignore_index=True)

    picks_file = os.path.join(output_dir, "picks.csv")
    df_picks.to_csv(picks_file, index=False)
    print(f"Saved {picks_file}: {len(df_picks)} picks (including controlled anomalies).")

    print("\nSummary of Raw Datasets Generated:")
    print(f"- orders.csv:  {len(df_orders):,} rows")
    print(f"- picks.csv:   {len(df_picks):,} rows")
    print(f"- items.csv:   {len(df_items):,} rows")
    print(f"- workers.csv: {len(df_workers):,} rows")
    print(f"- zones.csv:   {len(df_zones):,} rows")
    print("Controlled anomalies injected: 25 missing finish times, 35 inconsistent zone labels, "
          "5 negative durations, 12 missing workers, 18 duplicate rows.")

    return df_orders, df_picks, df_items, df_workers, df_zones


if __name__ == "__main__":
    generate_synthetic_data()
