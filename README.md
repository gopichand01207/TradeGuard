# 🛡️ TradeGuard – FinTech Behavioural Risk & Retention Engine

> *Resume project demonstrating end-to-end data engineering, ML modelling, and FinTech domain depth.*

---

## 📌 Project Summary

TradeGuard addresses the **93% retail F&O account churn benchmark** (SEBI, 2023) by:

| Capability | Details |
|---|---|
| **Data Engineering** | 76,000+ synthetic intraday F&O order logs (SQLite/Python) |
| **Behaviour Detection** | Revenge-trading & expiry-day over-leveraging flags |
| **Trader Segmentation** | K-Means (K=5) into 5 risk cohorts |
| **Risk Model** | Gradient Boosted drawdown classifier + regressor |
| **Real-Time Nudges** | GREEN / AMBER / RED risk signals per order |
| **Pre-Trade Kill-Switch** | Multi-rule + ML gate blocks high-risk orders |
| **A/B Simulator** | Monte-Carlo intervention analysis — 38% churn cut, 2.1x LTV |
| **Dashboard** | Interactive Streamlit app with Plotly visualisations |

---

## 🗂️ Project Structure

```
TradeGuard/
├── data/
│   ├── generate_data.py       # Synthetic order log generator (76k orders)
│   ├── tradeguard.db          # SQLite database (auto-generated)
│   └── orders.csv             # Raw orders CSV (auto-generated)
│
├── analysis/
│   ├── segmentation.py        # K-Means trader cohort segmentation
│   └── risk_model.py          # GBM risk classifier + drawdown regressor
│
├── simulation/
│   ├── kill_switch.py         # Pre-trade kill-switch logic
│   └── ab_simulator.py        # A/B intervention Monte-Carlo simulator
│
├── models/                    # Saved models & output CSVs (auto-generated)
├── plots/                     # All chart PNGs (auto-generated)
├── notebooks/                 # Jupyter notebooks (optional exploration)
│
├── app/
│   └── dashboard.py           # Streamlit interactive dashboard
│
├── run_pipeline.py            # One-command full pipeline runner
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (data → models → plots)
```bash
python run_pipeline.py
```

### 3. Launch the dashboard
```bash
streamlit run app/dashboard.py
```

The dashboard auto-detects missing artefacts and runs the pipeline on first launch.

---

## 📊 Key Metrics

| Metric | Value |
|---|---|
| Orders analysed | 76,000+ |
| Traders segmented | 500 |
| K-Means Silhouette Score | ~0.42 |
| Risk Classifier AUC-ROC | ~0.91 |
| Drawdown Regressor MAE | ~0.001 |
| Simulated churn reduction | **38%** |
| LTV uplift | **2.1×** |

---

## 🧠 Methodology

### Trader Segmentation (K-Means, K=5)
Features engineered per trader:
- **Revenge Rate** – orders placed within 10 min of a loss
- **Over-leverage Rate** – orders with leverage > 10×
- **Expiry Day Rate** – concentration on weekly/monthly expiry dates
- **Drawdown Intensity** – max vs. average drawdown ratio
- **Activity Density** – trading frequency relative to cohort

Optimal K selected via Elbow curve + Silhouette analysis.

### Risk Model (Gradient Boosting)
- **Classifier**: Predicts high-risk orders (drawdown > 5% threshold) — AUC ~0.91
- **Regressor**: Predicts exact drawdown % — R² ~ 0.60
- **Nudge Engine**: GREEN (safe) / AMBER (caution) / RED (block) per order

### Pre-Trade Kill-Switch
Multi-rule gate (any breach → block):
1. Leverage > 15× hard limit
2. Notional > configurable % of capital
3. Rapid-fire revenge trading (N orders in T minutes)
4. ML risk probability > 0.70
5. Expiry-day leverage > 8×

### A/B Intervention Simulator
- 1,000 Monte-Carlo bootstrap runs
- Control: no nudge, Treatment: full nudge stack
- Measured: churn rate, LTV months, avg PnL per trader

---

## 🛡️ Dashboard Pages

| Page | Description |
|---|---|
| 📊 Overview | KPIs, daily volume, PnL by archetype, leverage by symbol |
| 👥 Cohort Explorer | PCA scatter, radar chart, cohort drill-down table |
| ⚠️ Risk Nudge Monitor | Live nudge distribution, RED order heatmap |
| 🔴 Kill-Switch | Interactive order entry with real-time rule + ML evaluation |
| 🧪 A/B Simulator | Configurable Monte-Carlo simulation with live charts |
| 📈 Model Performance | All training plots (ROC, feature importance, regression) |

---

## 🏷️ Resume Bullet Mapping

> *"Addressed retail F&O account churn (93% SEBI loss benchmark) by analyzing 75,000+ intraday order logs (MySQL, Python) to detect revenge-trading and expiry-day over-leveraging."*

→ `data/generate_data.py` — 76,000 orders, SQLite (MySQL-compatible schema), revenge & expiry flags

> *"Segmented traders into 5 cohorts via K-Means and built a drawdown risk model for real-time risk nudges."*

→ `analysis/segmentation.py` + `analysis/risk_model.py`

> *"Built a pre-trade kill-switch and A/B intervention simulator (Streamlit), cutting simulated account blow-up churn by 38% and extending user LTV by 2.1x."*

→ `simulation/kill_switch.py` + `simulation/ab_simulator.py` + `app/dashboard.py`

---

## 📋 Tech Stack

- **Python 3.10+** — pandas, numpy, scikit-learn, scipy
- **Database** — SQLite (MySQL-compatible schema & queries)
- **ML** — scikit-learn (K-Means, GradientBoosting, PCA, StandardScaler)
- **Visualisation** — Plotly, Matplotlib, Seaborn
- **Dashboard** — Streamlit
- **Data Generation** — Faker (synthetic Indian trader profiles)

---

*Built for resume verification · © 2024 TradeGuard*

