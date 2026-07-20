# 🏪 Rossmann Store Sales Forecasting

**Forecasting daily store-level sales revenue for a 1,115-store European retail chain — with driver analysis, store segmentation, and a live deployed app.**

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://rossmann-sales-forecast-xyz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen)]()

🔗 **Live App:** [rossmann-sales-forecast-xyz.streamlit.app](https://rossmann-sales-forecast-xyz.streamlit.app/)
📓 **Full Notebook:** [`notebook/rossmann_analysis.ipynb`](notebook/rossmann_analysis.ipynb)

![App Screenshot](images/app_screenshot.png)

---

## 📌 The Problem

Rossmann operates 3,000+ drugstores across 7 European countries. Individual store managers currently forecast their own daily sales up to six weeks ahead — accuracy varies widely because it depends entirely on each manager's personal judgment.

**Goal:** Replace inconsistent manual forecasts with a data-driven model that predicts daily store-level sales, giving managers and planners a reliable, standardized tool to support staffing, cash-flow, and promotional decisions.

**Dataset:** [Rossmann Store Sales (Kaggle)](https://www.kaggle.com/c/rossmann-store-sales) — ~1M daily records, 1,115 stores, Jan 2013–Jul 2015.

---

## 🔍 Approach

| Step | What I did |
|---|---|
| **1. EDA** | Found strong weekly seasonality, near-zero Sunday sales (store closures), holiday spikes, and a **~38% average sales lift** during promotions |
| **2. Feature Engineering** | Built lag features (sales 7/14 days ago) and rolling averages (7/30-day trend) — carefully using `.shift(1)` before rolling windows to avoid leaking future data into training |
| **3. Modeling** | Benchmarked Linear Regression → Random Forest → XGBoost, using a **time-based train/test split** (last 6 weeks held out) since this is a forecasting problem, not a random classification task |
| **4. Driver Analysis** | Used XGBoost feature importance to identify what actually moves sales |
| **5. Store Segmentation** | Applied KMeans clustering on store-level sales behavior (avg. sales, volatility, promo sensitivity) to uncover distinct store archetypes |
| **6. Deployment** | Packaged the trained model into an interactive Streamlit app for live, on-demand predictions |

**Key modeling decisions I made deliberately (and can defend):**
- Dropped `Customers` as a feature — it's not known at prediction time, so using it would be data leakage
- Filtered out closed-store days (`Open=0`) before training — a closed store has €0 sales for structural reasons, not demand reasons
- Used a **time-based split**, not random — mimics how the model would actually be deployed (train on the past, predict the future)

---

## 📊 Results

| Model | MAE (€) | RMSE (€) |
|---|---|---|
| Linear Regression | 944.83 | 1,318.44 |
| Random Forest | 759.41 | 1,093.54 |
| **XGBoost (final)** | **668.39** | **963.88** |

- **Final error: ~9.6% of average daily sales** — comparable to competition-grade benchmarks on this exact dataset
- **Top 2 drivers** — a store's own recent 30-day sales trend and active promotions — together account for **~70% of predictive power**

![Feature Importance](images/feature_importance.png)

### Store Segmentation

KMeans clustering (k=4, chosen via elbow method) revealed 4 distinct store archetypes:

| Segment | Avg Sales | Promo Lift | # Stores | Profile |
|---|---|---|---|---|
| Flagship / High-Volume | €14,601 | €4,381 | 37 | Highest sales *and* most promo-responsive |
| Strong Performer | €8,768 | €3,697 | 168 | Above-average, promo-sensitive |
| Core / Average | €7,300 | €2,374 | 479 | Bulk of the network, middling on all dimensions |
| Steady, Low-Volume | €5,213 | €1,495 | 431 | Smallest, most stable, least promo-responsive |

![Cluster Profile](images/cluster_profile.png)

---

## 💡 Business Insight

Promotions are the strongest **controllable** lever for revenue — but their impact isn't uniform. The 37 flagship stores are both the highest-volume *and* the most promo-responsive, suggesting concentrated promotional spend there would yield outsized returns, while the 431 low-volume stores show minimal lift from promos. **A one-size-fits-all promotional strategy is leaving value on the table** — targeted, segment-aware promo budgeting would likely improve ROI.

---

## 🖥️ Try It Yourself

The deployed app lets you pick any store, a date, and promo/holiday conditions, and get an instant sales prediction — along with which behavioral segment that store belongs to.

👉 **[Launch the app](https://rossmann-sales-forecast-xyz.streamlit.app/)**

---

## 🛠️ Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `Streamlit` · `Matplotlib`

---

## 📁 Repo Structure

```
├── app.py                          # Streamlit app
├── requirements.txt
├── notebook/
│   └── rossmann_analysis.ipynb     # Full EDA → modeling → clustering workflow
├── models/
│   ├── xgb_model.pkl               # Trained XGBoost model
│   └── store_summary.pkl           # Store-level cluster profiles
├── data/
│   └── latest_store_state.csv      # Latest known rolling/lag features per store (used by app)
└── images/
    ├── feature_importance.png
    ├── cluster_profile.png
    └── app_screenshot.png
```

---

## ⚠️ Limitations & Future Work

- **Trained on 2013–2015 data** — would need periodic retraining on fresh data for reliable production use.
- **Store-level, not SKU-level** — the model forecasts total store revenue, not per-product demand, so it supports staffing/cash-flow/promotion decisions rather than inventory replenishment.
- **Live app uses last-known historical values** as a stand-in for recent sales trend (rather than a live database connection) — a reasonable simplification for a portfolio demo, clearly noted in the app itself.
- **Next steps:** incorporate weather and local event data, model competitor tenure (`CompetitionOpenSince`), and properly encode `Promo2` cycle timing (currently dropped for simplicity).

---

## 👤 Author

**Krishna Agrawal** — B.Tech Mechanical Engineering, MNNIT Allahabad
[GitHub](https://github.com/krishnaagrawal74) · [LinkedIn](#)