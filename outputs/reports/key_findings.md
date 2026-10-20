# Executive Report: Warehouse Efficiency Insight (Round 2 Findings)
**Project:** CH22 — Warehouse Efficiency Insight  
**Organization:** FlowRack Logistics Pvt. Ltd.  
**Sector:** Supply Chain & Logistics Operations  
**Assessment:** DS Sprint — Skill Based Assessment 2 | Round 2: EDA & Feature Engineering  
**Prepared By:** Senior Data Science Lead  
**Dataset Provenance:** Reproducible Synthetic Operations Model (Seed: 42)  

---

## 1. Executive Summary
FlowRack Logistics operates high-throughput multi-zone fulfillment centers. In fulfillment operations, **order picking** represents 55% to 65% of warehouse operational labor costs and serves as the principal constraint on delivery turnaround speed.

This exploratory data analysis and feature engineering study investigates:
> **"What factors contribute to longer picking and order-processing times?"**

Through end-to-end data auditing, systematic cleaning, grain-aware dataset integration, statistical hypothesis testing (ANOVA, Kruskal-Wallis, Pearson correlation), and multi-method feature selection, we analyzed **5,348 valid pick events** spanning **999 completed customer orders**, 120 SKU catalog items, 40 associates, and 6 specialized warehouse zones.

---

## 2. Key Empirical Findings (By Business Question)

### Question 1: Which Warehouse Zones Take Longer?
- **Empirical Results:**
  - **Zone F (Bulk Storage, Deep-Storage):** Mean = **13.52 minutes** (Median = 13.56 min, Std = 2.92 min).
  - **Zone E (Cold Vault, Far):** Mean = **11.23 minutes** (Median = 11.27 min, Std = 2.41 min).
  - **Zone D (High-Rack, Mid):** Mean = **8.86 minutes** (Median = 8.84 min, Std = 1.94 min).
  - **Zone C (Mezzanine, Mid):** Mean = **7.84 minutes** (Median = 7.82 min, Std = 1.70 min).
  - **Zone B (Standard Shelf, Near):** Mean = **5.58 minutes** (Median = 5.57 min, Std = 1.25 min).
  - **Zone A (Fast-Moving, Near):** Mean = **4.21 minutes** (Median = 4.19 min, Std = 0.96 min).
- **Statistical Significance:** One-way ANOVA across zones confirms strong statistical divergence:  
  **$F = 1,886.94, \; p < 10^{-300}$**; Kruskal-Wallis **$H = 3,616.30, \; p < 10^{-300}$**.
- **Root Drivers:**
  1. **Facility Travel Distance:** Distance score is strongly correlated with picking time ($r = 0.68$). Deep-storage racks require protracted walk/cart transit from packing docks.
  2. **Access Complexity:** Bulk storage requires pallet jacks and forklift staging, while Cold Vault requires thermal gear and airlock door operations.

---

### Question 2: How Does Item Count Affect Picking Time?
- **Empirical Results:**
  - **Small Orders (1–3 items):** Average **18.7 elapsed minutes**; Mean labor duration sum = **16.8 minutes**.
  - **Medium Orders (4–7 items):** Average **51.2 elapsed minutes**; Mean labor duration sum = **45.6 minutes**.
  - **Large Orders (8–12 items):** Average **94.8 elapsed minutes**; Mean labor duration sum = **82.4 minutes**.
  - **Bulk Orders (13+ items):** Average **145.2 elapsed minutes**; Mean labor duration sum = **118.4 minutes**.
- **Statistical Relationship:**
  - Order item count displays near-perfect linear scaling with total elapsed turnaround time:  
    **Pearson correlation $r = 0.970, \; p < 10^{-300}$**.
- **Critical Grain Distinction:**
  - Individual pick duration does **not** change dramatically with order size (~8.2 minutes per pick across all order types).
  - However, aggregate order turnaround expands linearly due to sequential routing across multiple storage zones.

---

### Question 3: Which Shifts Experience Delays?
- **Empirical Results:**
  - **Morning Shift (06:00 – 14:00):** Mean = **8.07 minutes** (Median = 7.78 min, Std = 3.86 min).
  - **Afternoon Shift (14:00 – 22:00):** Mean = **8.31 minutes** (Median = 8.11 min, Std = 3.97 min).
  - **Night Shift (22:00 – 06:00):** Mean = **8.84 minutes** (Median = 8.65 min, Std = 4.18 min).
- **Statistical Significance:** One-way ANOVA indicates a statistically significant difference:  
  **$F = 9.60, \; p = 6.89 \times 10^{-5}$**.
- **Operational Reality:**
  - Night shift pick duration is **9.5% higher** than Morning shift.
  - Rather than associate negligence, this reflects circadian fatigue, reduced floor supervision, and fewer forklift spotters active in high-rack aisles during overnight hours.

---

### Question 4: How Does Worker Experience Relate to Picking Time?
- **Empirical Results:**
  - **Senior Associates (> 3 years):** Mean = **6.97 minutes** (Median = 6.64 min, Std = 3.38 min).
  - **Intermediate Associates (1–3 years):** Mean = **8.04 minutes** (Median = 7.85 min, Std = 3.79 min).
  - **Novice Associates (< 1 year):** Mean = **9.49 minutes** (Median = 9.38 min, Std = 4.31 min).
- **Statistical Significance:** One-way ANOVA across tenure tiers:  
  **$F = 169.49, \; p = 4.32 \times 10^{-72}$**.
- **Tenure Differential:**
  - Senior workers execute picks **26.5% faster** than Novices (saving an average of **2.52 minutes per pick action**).
  - Senior associates exhibit superior spatial awareness of bin subdivisions and barcode scanning proficiency.

---

### Question 5: Which Item Categories Slow Down Picking?
- **Empirical Results:**
  - **Heavy Equipment:** Mean = **13.56 minutes** (Bulky, Deep-Storage/Hazardous).
  - **Pharmaceuticals:** Mean = **9.81 minutes** (Cold Storage / Secure High-Value).
  - **Home Goods:** Mean = **8.24 minutes** (Medium/Large, Ambient).
  - **Groceries:** Mean = **7.59 minutes** (Small/Medium, Ambient & Cold).
  - **Electronics:** Mean = **7.11 minutes** (Small/Medium, Secure High-Value).
  - **Apparel:** Mean = **5.24 minutes** (Small/Medium, Ambient, Fast-Moving).
- **Statistical Significance:** One-way ANOVA across categories:  
  **$F = 473.61, \; p < 10^{-300}$**.
- **Composite Handling Complexity:** The composite index combining physical size and storage rules shows a **0.64 correlation** with pick duration.

---

### Question 6: Where Does Queue Build-up Occur?
- **Queue Definition:** Elapsed time between order arrival/release (`order_timestamp`) and first pick start (`first_pick_start`).
- **Empirical Results:**
  - **Urgent Orders:** Mean = **12.4 minutes** (Max: 19.8 min).
  - **High Priority:** Mean = **27.6 minutes** (Max: 39.9 min).
  - **Medium Priority:** Mean = **42.3 minutes** (Max: 59.8 min).
  - **Low Priority:** Mean = **67.3 minutes** (Max: 94.6 min).
- **Dispatch Adherence:** Warehouse wave-release logic strictly prioritizes high-SLA orders ($F = 2,410.8, p < 10^{-300}$).
- **Queue Limitations:** Available telemetry captures order release latency, but does not capture intra-aisle picker cart congestion.

---

## 3. Feature Selection Summary
Target: `picking_duration_minutes` (leakage-free baseline).
Evaluating Mutual Information, Random Forest MDI, and LassoCV:

| Rank | Feature | Mutual Info | RF Importance % | Lasso Abs Coef | Decision |
|:---:|:---|:---:|:---:|:---:|:---:|
| 1 | `zone_difficulty_score` | 0.4412 | 34.2% | 1.4210 | **Selected** |
| 2 | `handling_complexity_score`| 0.3891 | 28.5% | 1.1540 | **Selected** |
| 3 | `distance_score` | 0.3620 | 18.4% | 0.9820 | **Selected** |
| 4 | `item_size_score` | 0.3105 | 8.1% | 0.6430 | **Selected** |
| 5 | `experience_score` | 0.1840 | 5.2% | 0.4910 | **Selected** |
| 6 | `picking_quantity` | 0.1250 | 3.1% | 0.3120 | **Selected** |
| 7 | `storage_class_score` | 0.1420 | 1.8% | 0.2850 | **Selected** |
| 8 | `pick_hour` | 0.0410 | 0.7% | 0.0820 | Excluded |
| 9 | `order_priority_score` | 0.0120 | 0.0% | 0.0000 | Excluded |

---

## 4. Strategic Warehouse Recommendations
1. **Dynamic Slotting Reallocation:** Reallocate high-velocity items from Zone F/E closer to intermediate access docks.
2. **Mentorship & Zone Pairing:** Pair Novice pickers with Senior associates or assign Novices to Zone A/B during initial 60 days.
3. **Multi-Picker Batch Wave Dispatching:** For orders with >8 items, dispatch parallel pick waves across zones rather than single-worker sequential walk routes.
4. **Night Shift Material Support:** Provide dedicated forklift spotters during 22:00–06:00 to reduce high-rack wait times.
