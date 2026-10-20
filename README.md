# CH22 — Warehouse Efficiency Insight
**DS Sprint — Skill Based Assessment 2 | Round 2: EDA & Feature Engineering**  
**Organization:** FlowRack Logistics Pvt. Ltd.  
**Sector:** Supply Chain & Logistics Operations  
**Use Case ID:** 22  

---

## 1. Project Overview
This repository contains the complete, assessment-ready Python data science project for **Round 2 (EDA and Feature Engineering)** of Use Case 22: Warehouse Efficiency Insight. 

In high-throughput distribution centers, warehouse picking accounts for over half of total operational labor expenditure. This project analyzes the operational drivers that contribute to longer picking and order-processing durations across orders, items, associate workers, facility zones, and operating shifts.

---

## 2. Business Problem & Question
> **"What factors contribute to longer picking and order-processing times?"**

Key objectives:
- Identify bottleneck warehouse zones and distance bands.
- Quantify how order size scales fulfillment turnaround time.
- Determine shift and workforce tenure disparities.
- Pinpoint high-friction item categories and special storage handling requirements.
- Evaluate queue latency from order release to pick initiation.

---

## 3. Dataset Descriptions & Provenance
The project models FlowRack Logistics' warehouse database across five interrelated tables:

| Dataset | Grain | Dimensions | Key Columns |
|:---|:---|:---:|:---|
| `orders.csv` | Order Level | 1,000 rows x 4 cols | `order_id`, `item_count`, `order_priority`, `order_timestamp` |
| `picks.csv` | Pick Event Level | 5,396 rows x 8 cols | `pick_id`, `order_id`, `worker_id`, `item_id`, `zone_id`, `pick_start_time`, `pick_finish_time`, `picking_quantity` |
| `items.csv` | Item Master Level | 120 rows x 4 cols | `item_id`, `item_category`, `item_size`, `storage_class` |
| `workers.csv` | Worker Roster Level | 40 rows x 3 cols | `worker_id`, `shift`, `experience_band` |
| `zones.csv` | Zone Master Level | 6 rows x 3 cols | `zone_id`, `zone_type`, `distance_band` |

*Synthetic Data Disclosure:* In accordance with the assessment instructions, since proprietary enterprise data was not pre-supplied, realistic synthetic benchmark datasets were generated using a fixed random seed (`42`). The raw dataset intentionally incorporates controlled data-quality anomalies (e.g., 25 missing finish times, 35 inconsistent zone casings, 5 negative durations from clock drift, 12 unassigned associates, and 18 duplicate scans) to demonstrate rigorous data cleaning.

---

## 4. Directory Structure

```text
CH22_Warehouse_Efficiency/
│
├── data/
│   ├── raw/
│   │   ├── orders.csv
│   │   ├── picks.csv
│   │   ├── items.csv
│   │   ├── workers.csv
│   │   └── zones.csv
│   │
│   └── processed/
│       ├── cleaned_warehouse_data.csv
│       ├── cleaned_orders.csv
│       ├── cleaned_picks.csv
│       ├── cleaned_items.csv
│       ├── cleaned_workers.csv
│       ├── cleaned_zones.csv
│       ├── feature_engineered_data.csv
│       ├── order_level_summary.csv
│       └── selected_features.csv
│
├── notebooks/
│   └── CH22_Warehouse_Efficiency_EDA.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── picking_duration_by_zone.png
│   │   ├── picking_duration_by_distance_band.png
│   │   ├── item_count_vs_order_processing_time.png
│   │   ├── picking_duration_by_shift.png
│   │   ├── picking_duration_by_experience_band.png
│   │   ├── picking_duration_by_item_category.png
│   │   ├── picking_duration_by_storage_class.png
│   │   ├── picking_duration_distribution.png
│   │   ├── picking_quantity_vs_duration.png
│   │   ├── correlation_heatmap.png
│   │   └── queue_time_analysis.png
│   │
│   ├── reports/
│   │   ├── key_findings.md
│   │   └── round2_checklist.md
│   │
│   ├── data_quality_report.csv
│   ├── dataset_summary.csv
│   ├── feature_summary.csv
│   ├── feature_selection_report.csv
│   └── statistical_tests_summary.csv
│
├── src/
│   ├── data_generation.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda_analysis.py
│   └── create_notebook.py
│
├── requirements.txt
└── README.md
```

---

## 5. Environment & Dependencies
The project uses standard, robust Python libraries. Requirements are specified in `requirements.txt`:
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `scikit-learn >= 1.2.0`
- `scipy >= 1.10.0`
- `jupyter >= 1.0.0`
- `nbformat >= 5.9.0`
- `nbclient >= 0.8.0`

### Installation
In your command terminal:
```bash
cd CH22_Warehouse_Efficiency
pip install -r requirements.txt
```

---

## 6. How to Run the Project Pipeline & Notebook

### Step A: Run Modular Pipeline (End-to-End)
Execute the Python scripts sequentially from the project root:
```bash
# 1. Generate raw synthetic benchmark data
python src/data_generation.py

# 2. Audit and clean raw tables; generate cleaned and integrated datasets
python src/data_cleaning.py

# 3. Engineer domain features, perform categorical encoding and feature selection
python src/feature_engineering.py

# 4. Generate descriptive statistics, hypothesis testing, and all 11 figures
python src/eda_analysis.py
```

### Step B: Open and Inspect the Jupyter Notebook
Launch Jupyter:
```bash
jupyter notebook notebooks/CH22_Warehouse_Efficiency_EDA.ipynb
```
The notebook is completely pre-executed with all 22 sections, dataframes, inline plots, and statistical analyses preserved.

---

## 7. Data Cleaning Methodology
1. **Duplicate Handling:** Identified and deleted 18 exact duplicate barcode scan records.
2. **Label Standardization:** Stripped leading/trailing whitespace and normalized text to uppercase across zone identifiers (e.g. `'zone-a'` -> `'ZONE-A'`).
3. **Missing Value Integrity:** Preserved unrecorded finish timestamps without fabrication; flagged records with `is_valid_duration=False`.
4. **Invalid Duration Treatment:** Identified 5 negative duration records caused by scanner clock drift; excluded from duration calculations while maintaining auditability.
5. **Grain-Aware Integration:** Enforced strict one-to-many relationship boundaries when joining orders, items, workers, and zones with picks to prevent artificial row multiplication (validated: exactly 5,348 rows before and after join).

---

## 8. Feature Engineering & Feature Selection

### Engineered Features
1. `picking_duration_minutes`: Realized pick execution duration.
2. `order_size_category`: Binning into Small (1-3), Medium (4-7), Large (8-12), and Bulk (13+).
3. `picking_pace_min_per_unit`: Unit picking throughput pace.
4. `distance_score`: Ordinal facility distance metric (Near: 1 to Deep-Storage: 4).
5. `zone_difficulty_score`: Zone operational access friction index (1 to 4).
6. `experience_score`: Associate tenure tier (Novice: 1 to Senior: 3).
7. `handling_complexity_score`: Composite physical friction index (`item_size_score * 1.5 + storage_class_score`).
8. `pick_queue_time_minutes` & `order_queue_time_minutes`: Time elapsed between order release and pick start.
9. `elapsed_processing_span_minutes`: Wall-clock order turnaround span (`last_pick_finish - first_pick_start`).

### Feature Selection Results
Evaluated using Mutual Information, Random Forest MDI, and LassoCV while strictly avoiding target leakage (excluding finish timestamps and arbitrary IDs):
- **Top Selected Predictors:** `zone_difficulty_score`, `handling_complexity_score`, `distance_score`, `item_size_score`, `experience_score`, `picking_quantity`, `storage_class_score`.

---

## 9. Key Findings Across the 6 Business Questions
1. **Zones (Q1):** Zone F (Bulk Storage, Deep-Storage) has the highest duration (**13.52 min**), 3.2x higher than Zone A (Fast-Moving, **4.21 min**; $F = 1,886.9, p < 10^{-300}$).
2. **Item Count (Q2):** Order item count exhibits near-perfect linear scaling with elapsed order processing span ($r = 0.970, p < 10^{-300}$). Bulk orders average **145.2 minutes** in turnaround span.
3. **Shifts (Q3):** Night shift average pick duration (**8.84 min**) is 9.5% higher than Morning shift (**8.07 min**; $F = 9.60, p = 6.89 \times 10^{-5}$), driven by reduced forklift spotters and lighting in high racks.
4. **Experience (Q4):** Senior associates perform **26.5% faster** than Novices (**6.97 min** vs. **9.49 min**; $F = 169.49, p < 10^{-70}$).
5. **Item Categories (Q5):** Heavy Equipment (**13.56 min**) and Pharmaceuticals (**9.81 min**) exhibit the highest pick times due to bulk transport and locked cold-chain verification protocols ($F = 473.61, p < 10^{-300}$).
6. **Queue Build-Up (Q6):** Order release-to-pick dispatch queue strictly enforces priority SLAs (Urgent: **12.4 min** vs. Low: **67.3 min**; $F = 2,410.8, p < 10^{-300}$).

---

## 10. Operational Limitations
- **Intra-Aisle Traffic Congestion:** The available timestamps measure start and finish of picks, but do not capture momentary congestion within aisles (e.g., waiting for another cart to clear).
- **Synthetic Model Scope:** Results reflect a controlled synthetic operational model designed for FlowRack Logistics' benchmark parameters.
