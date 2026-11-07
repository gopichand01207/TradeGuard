"""
TradeGuard – Streamlit Dashboard
Behavioural Risk & Retention Engine for Retail F&O Traders
"""

import sys, os
# Make sibling directories importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeGuard | F&O Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── paths ─────────────────────────────────────────────────────────────────────
DB_PATH    = os.path.join(ROOT, "data",   "tradeguard.db")
MODEL_DIR  = os.path.join(ROOT, "models")
PLOTS_DIR  = os.path.join(ROOT, "plots")

COHORT_LABELS = {
    0: "Revenge Traders",
    1: "Expiry Gamblers",
    2: "Disciplined Traders",
    3: "Over-Leveraged Newbies",
    4: "Steady Losers",
}
COHORT_COLORS = ["#e74c3c","#e67e22","#2ecc71","#9b59b6","#3498db"]

FEATURES = [
    "quantity", "leverage_ratio", "notional_pct_capital",
    "is_expiry_day", "revenge_flag", "overleveraged",
    "hour", "minute", "day_of_week",
    "experience_months", "risk_score",
]

# ── data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_db():
    conn     = sqlite3.connect(DB_PATH)
    orders   = pd.read_sql("SELECT * FROM orders",         conn)
    traders  = pd.read_sql("SELECT * FROM traders",        conn)
    summary  = pd.read_sql("SELECT * FROM trader_summary", conn)
    conn.close()
    return orders, traders, summary


@st.cache_resource
def load_models():
    models = {}
    for name, fname in [
        ("clf",    "risk_classifier.pkl"),
        ("reg",    "drawdown_regressor.pkl"),
        ("km",     "kmeans_model.pkl"),
        ("scaler", "scaler.pkl"),
    ]:
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)
    return models


@st.cache_data(ttl=300)
def load_cohorts():
    p = os.path.join(MODEL_DIR, "trader_cohorts.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def setup_check():
    """Return True only when all required artefacts exist."""
    required = [
        os.path.join(ROOT, "data", "tradeguard.db"),
        os.path.join(MODEL_DIR, "risk_classifier.pkl"),
        os.path.join(MODEL_DIR, "trader_cohorts.csv"),
    ]
    return all(os.path.exists(p) for p in required)


# ── sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.image("https://img.shields.io/badge/TradeGuard-F%26O%20Risk%20Engine-blue?style=for-the-badge",
                 use_container_width=True)
        st.markdown("### 🛡️ TradeGuard")
        st.markdown("*Behavioural Risk & Retention Engine*")
        st.divider()

        page = st.radio("Navigate", [
            "📊 Dashboard Overview",
            "👥 Trader Cohort Explorer",
            "⚠️ Risk Nudge Monitor",
            "🔴 Pre-Trade Kill-Switch",
            "🧪 A/B Intervention Simulator",
            "📈 Model Performance",
        ])
        st.divider()
        st.caption("SEBI Data: 93% retail F&O traders lose")
        st.caption("Dataset: 76,000+ synthetic intraday orders")
        st.caption("© 2024 TradeGuard · Resume Project")
    return page


# ── pages ─────────────────────────────────────────────────────────────────────

def page_overview(orders, traders, summary):
    st.title("📊 TradeGuard – Dashboard Overview")
    st.markdown("""
    > **TradeGuard** is a FinTech Behavioural Risk & Retention Engine that analyzes  
    > retail F&O order logs to detect revenge-trading & expiry-day over-leveraging,  
    > segments traders into risk cohorts, and deploys real-time risk nudges.
    """)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders",    f"{len(orders):,}")
    c2.metric("Active Traders",  f"{len(traders):,}")
    c3.metric("Revenge Trades",  f"{orders['revenge_flag'].sum():,}")
    c4.metric("Overleveraged",   f"{orders['overleveraged'].sum():,}")
    c5.metric("Expiry Day Orders", f"{orders['is_expiry_day'].sum():,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Daily Order Volume")
        daily = orders.groupby("trade_date").size().reset_index(name="orders")
        daily["trade_date"] = pd.to_datetime(daily["trade_date"])
        fig = px.area(daily, x="trade_date", y="orders",
                      color_discrete_sequence=["#3498db"],
                      labels={"trade_date":"Date","orders":"Orders"})
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 PnL Distribution by Archetype")
        merged = orders.merge(traders[["trader_id","archetype"]], on="trader_id")
        fig = px.box(merged, x="archetype", y="pnl",
                     color="archetype",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("⚡ Revenge Trade Rate by Archetype")
        merged["revenge"] = merged["revenge_flag"]
        rv = merged.groupby("archetype")["revenge"].mean().reset_index()
        rv.columns = ["archetype","revenge_rate"]
        fig = px.bar(rv, x="archetype", y="revenge_rate",
                     color="archetype",
                     color_discrete_sequence=px.colors.qualitative.Set1,
                     labels={"revenge_rate":"Revenge Trade Rate"})
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%",
                          margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("📊 Average Leverage by Symbol")
        sym = orders.groupby("symbol")["leverage_ratio"].mean().reset_index()
        fig = px.bar(sym, x="symbol", y="leverage_ratio",
                     color="symbol",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     labels={"leverage_ratio":"Avg Leverage Ratio"})
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)


def page_cohort_explorer(summary, traders):
    st.title("👥 Trader Cohort Explorer")
    cohorts = load_cohorts()
    if cohorts is None:
        st.warning("Run segmentation first (analysis/segmentation.py)")
        return

    cohorts["cohort_name"] = cohorts["cohort"].map(COHORT_LABELS)

    # PCA scatter – compute before merge so cohorts stays clean
    if "PC1" not in cohorts.columns:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        feat_cols = ["revenge_rate","overlev_rate","expiry_rate",
                     "avg_leverage","pnl_per_trade","max_drawdown",
                     "drawdown_intensity","activity_density"]
        X = cohorts[feat_cols].fillna(0).values
        sc = StandardScaler()
        X_s = sc.fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        pc  = pca.fit_transform(X_s)
        cohorts["PC1"] = pc[:,0]
        cohorts["PC2"] = pc[:,1]
        cohorts["cohort_name"] = cohorts["cohort"].map(COHORT_LABELS)

    # Drop 'capital' from cohorts before merge to avoid _x/_y suffix collision
    # (trader_summary already carries capital; we re-join from traders for name/city)
    cohorts_for_merge = cohorts.drop(columns=["capital"], errors="ignore")
    merged = cohorts_for_merge.merge(
        traders[["trader_id","name","city","capital"]], on="trader_id", how="left"
    )

    st.subheader("PCA – Trader Cohort Scatter")
    fig = px.scatter(cohorts, x="PC1", y="PC2", color="cohort_name",
                     color_discrete_sequence=COHORT_COLORS,
                     hover_data={"cohort_name":True,"total_pnl":True},
                     labels={"cohort_name":"Cohort"})
    fig.update_traces(marker=dict(size=8, opacity=0.75))
    fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cohort KPIs
    st.subheader("Cohort Behavioral Summary")
    kpi_cols = ["revenge_rate","overlev_rate","expiry_rate",
                "avg_leverage","max_drawdown","total_pnl"]
    grp = cohorts.groupby("cohort_name")[kpi_cols].mean().reset_index()
    grp.columns = ["Cohort","Revenge Rate","Overlev Rate","Expiry Rate",
                   "Avg Leverage","Max Drawdown","Avg PnL"]
    st.dataframe(grp.style.format({
        "Revenge Rate":  "{:.2%}",
        "Overlev Rate":  "{:.2%}",
        "Expiry Rate":   "{:.2%}",
        "Avg Leverage":  "{:.1f}x",
        "Max Drawdown":  "{:.2f}%",
        "Avg PnL":       "₹{:,.0f}",
    }), use_container_width=True)

    # Cohort size pie
    st.subheader("Cohort Size Distribution")
    sizes = cohorts["cohort_name"].value_counts().reset_index()
    sizes.columns = ["Cohort","Count"]
    fig2 = px.pie(sizes, names="Cohort", values="Count",
                  color_discrete_sequence=COHORT_COLORS, hole=0.4)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # Filter by cohort
    st.divider()
    st.subheader("🔍 Drill Into a Cohort")
    sel = st.selectbox("Select Cohort", list(COHORT_LABELS.values()))
    cid = {v:k for k,v in COHORT_LABELS.items()}[sel]
    sub = merged[merged["cohort"] == cid][
        ["trader_id","name","city","capital","total_orders",
         "total_pnl","avg_leverage","revenge_trades","overleveraged_trades"]
    ].sort_values("total_pnl")
    st.dataframe(sub.style.format({
        "capital":    "₹{:,.0f}",
        "total_pnl":  "₹{:,.0f}",
        "avg_leverage": "{:.1f}x",
    }), use_container_width=True, height=350)


def page_risk_nudge(orders, traders):
    st.title("⚠️ Real-Time Risk Nudge Monitor")

    nudge_path = os.path.join(MODEL_DIR, "orders_with_nudges.csv")
    if not os.path.exists(nudge_path):
        st.warning("Run risk model first (analysis/risk_model.py)")
        return

    df = pd.read_csv(nudge_path, low_memory=False)

    # KPIs
    total  = len(df)
    green  = (df["nudge_level"] == "GREEN 🟢").sum()
    amber  = (df["nudge_level"] == "AMBER 🟡").sum()
    red    = (df["nudge_level"] == "RED 🔴").sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Orders",   f"{total:,}")
    c2.metric("🟢 GREEN",       f"{green:,}",  f"{green/total:.1%}")
    c3.metric("🟡 AMBER",       f"{amber:,}",  f"{amber/total:.1%}")
    c4.metric("🔴 RED (Block)", f"{red:,}",    f"{red/total:.1%}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nudge Distribution")
        nudge_counts = df["nudge_level"].value_counts().reset_index()
        nudge_counts.columns = ["Nudge","Count"]
        fig = px.bar(nudge_counts, x="Nudge", y="Count",
                     color="Nudge",
                     color_discrete_map={
                         "GREEN 🟢":"#2ecc71",
                         "AMBER 🟡":"#f39c12",
                         "RED 🔴":"#e74c3c",
                     })
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Probability Distribution")
        fig2 = px.histogram(df, x="risk_prob", nbins=50,
                            color_discrete_sequence=["#9b59b6"],
                            labels={"risk_prob":"ML Risk Score"})
        fig2.add_vline(x=0.33, line_dash="dash", line_color="#f39c12",
                       annotation_text="AMBER threshold")
        fig2.add_vline(x=0.66, line_dash="dash", line_color="#e74c3c",
                       annotation_text="RED threshold")
        fig2.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # High-risk orders table
    st.subheader("🔴 High-Risk Orders Requiring Intervention")
    red_orders = df[df["nudge_level"] == "RED 🔴"].nlargest(50, "risk_prob")[
        ["order_id","trader_id","symbol","side","quantity",
         "leverage_ratio","drawdown_pct","risk_prob","revenge_flag","is_expiry_day"]
    ]
    st.dataframe(red_orders.style.format({
        "risk_prob":     "{:.4f}",
        "leverage_ratio":"{:.2f}x",
        "drawdown_pct":  "{:.4f}%",
    }).background_gradient(subset=["risk_prob"], cmap="Reds"),
    use_container_width=True, height=350)


def page_kill_switch():
    st.title("🔴 Pre-Trade Kill-Switch Simulator")
    st.markdown("""
    Simulate real-time order evaluation against the multi-rule risk gate.
    Any breach → order **blocked** before reaching the exchange.
    """)

    st.subheader("⚙️ Kill-Switch Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        max_lev   = st.slider("Max Leverage (x)",       5.0, 30.0, 15.0, 0.5)
        ml_thresh = st.slider("ML Risk Threshold",      0.3, 0.95, 0.70, 0.05)
    with col2:
        max_dd    = st.slider("Max Drawdown % of Cap",  2.0, 20.0, 8.0,  0.5)
        rev_win   = st.slider("Revenge Window (min)",   5,   30,   10)
    with col3:
        rev_max   = st.slider("Max Orders in Window",   2,   15,   5)
        exp_lev   = st.slider("Expiry Day Max Lev (x)", 3.0, 15.0, 8.0, 0.5)

    st.divider()
    st.subheader("📋 Manual Order Entry")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        symbol   = st.selectbox("Symbol", ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY","SENSEX"])
        quantity = st.number_input("Quantity (lots × lot_size)", min_value=25, value=500, step=25)
        price    = st.number_input("Entry Price (₹)", min_value=1.0, value=22000.0, step=50.0)
    with col_b:
        capital  = st.number_input("Account Capital (₹)", min_value=10000, value=200000, step=10000)
        is_exp   = st.checkbox("Is Expiry Day?", value=False)
        revenge  = st.checkbox("Rapid-fire order (revenge flag)?", value=False)
    with col_c:
        exp_months = st.number_input("Experience (months)", 1, 120, 12)
        risk_tol   = st.selectbox("Risk Tolerance", ["low","medium","high"])
        n_recent   = st.number_input("Recent orders in window", 0, 20, 0)

    if st.button("🔍 Evaluate Order", type="primary"):
        notional  = price * quantity
        leverage  = round(notional / max(capital, 1), 2)
        overlev   = int(leverage > 10)
        lot_map   = {"NIFTY":25,"BANKNIFTY":15,"FINNIFTY":40,"MIDCPNIFTY":75,"SENSEX":10}
        lot_size  = lot_map.get(symbol, 25)

        order = {
            "quantity":       quantity,
            "leverage_ratio": leverage,
            "notional_value": notional,
            "is_expiry_day":  int(is_exp),
            "revenge_flag":   int(revenge),
            "overleveraged":  overlev,
            "order_time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        trader = {
            "capital":           capital,
            "experience_months": exp_months,
            "risk_tolerance":    risk_tol,
        }

        # Inline rule-based evaluation (no model needed)
        reasons = []
        if leverage > max_lev:
            reasons.append(f"⛔ Leverage {leverage:.1f}x > limit {max_lev}x")
        if notional / capital * 100 > max_dd * 10:
            reasons.append(f"⛔ Notional {notional/capital*100:.1f}% of capital > threshold")
        if n_recent >= rev_max:
            reasons.append(f"⛔ Rapid-fire: {n_recent} recent orders ≥ limit {rev_max}")
        if is_exp and leverage > exp_lev:
            reasons.append(f"⛔ Expiry-day leverage {leverage:.1f}x > {exp_lev}x hard-limit")

        # ML risk score if model available
        models = load_models()
        risk_prob = 0.0
        if "clf" in models:
            risk_map = {"low":0,"medium":1,"high":2}
            now = datetime.now()
            feat = [
                quantity, leverage, notional/max(capital,1)*100,
                int(is_exp), int(revenge), overlev,
                now.hour, now.minute, now.weekday(),
                exp_months, risk_map.get(risk_tol, 1),
            ]
            risk_prob = float(models["clf"].predict_proba([feat])[0][1])
            if risk_prob > ml_thresh:
                reasons.append(f"⛔ ML risk score {risk_prob:.3f} > threshold {ml_thresh}")

        blocked = len(reasons) > 0
        nudge   = "🔴 RED" if risk_prob > 0.66 else "🟡 AMBER" if risk_prob > 0.33 else "🟢 GREEN"

        st.divider()
        if blocked:
            st.error(f"### ❌ ORDER BLOCKED  |  {nudge}")
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.success(f"### ✅ ORDER APPROVED  |  {nudge}")

        col_x, col_y, col_z = st.columns(3)
        col_x.metric("Notional Value",   f"₹{notional:,.0f}")
        col_y.metric("Leverage Ratio",   f"{leverage:.1f}x")
        col_z.metric("ML Risk Score",    f"{risk_prob:.3f}")


def page_ab_simulator():
    st.title("🧪 A/B Intervention Simulator")
    st.markdown("""
    Monte-Carlo simulation of **risk nudge intervention** vs. control group.
    Models the SEBI benchmark of 93% retail F&O loss rate.
    """)

    col1, col2 = st.columns(2)
    with col1:
        n_traders   = st.slider("Sample Size (traders per arm)", 100, 1000, 500, 50)
        ctrl_churn  = st.slider("Control Group Churn Rate",      0.4, 0.95, 0.72, 0.01)
        nudge_effect= st.slider("Nudge Effect (churn reduction)",0.10, 0.60, 0.38, 0.01)
    with col2:
        ltv_base    = st.slider("Baseline LTV (months)",         3, 24, 8)
        ltv_mult    = st.slider("LTV Multiplier",                1.0, 4.0, 2.1, 0.1)
        n_sims      = st.slider("Monte-Carlo Simulations",       100, 2000, 1000, 100)

    if st.button("▶️ Run Simulation", type="primary"):
        sys.path.insert(0, os.path.join(ROOT, "simulation"))
        from simulation.ab_simulator import simulate_ab_test

        with st.spinner("Running Monte-Carlo …"):
            stats = simulate_ab_test(
                n_traders=n_traders,
                control_churn_rate=ctrl_churn,
                nudge_effect=nudge_effect,
                ltv_base_months=ltv_base,
                ltv_multiplier=ltv_mult,
                n_simulations=n_sims,
            )

        st.divider()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Control Churn",   f"{stats['control_churn']:.2%}")
        c2.metric("Treatment Churn", f"{stats['treatment_churn']:.2%}",
                  delta=f"-{stats['churn_reduction']:.1%}", delta_color="inverse")
        c3.metric("Control LTV",     f"{stats['ctrl_ltv_months']}mo")
        c4.metric("Treatment LTV",   f"{stats['trt_ltv_months']:.1f}mo",
                  delta=f"+{ltv_mult-1:.1%}")

        col_a, col_b = st.columns(2)
        with col_a:
            ctrl_arr = np.array(stats["control_churns"])
            trt_arr  = np.array(stats["treatment_churns"])
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ctrl_arr, name="Control",
                                       marker_color="#e74c3c", opacity=0.7))
            fig.add_trace(go.Histogram(x=trt_arr,  name="Treatment",
                                       marker_color="#2ecc71", opacity=0.7))
            fig.update_layout(
                barmode="overlay", title="Churn Rate Distributions",
                xaxis_title="Churn Rate", yaxis_title="Frequency",
                margin=dict(l=0,r=0,t=30,b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            labels = ["SEBI Benchmark<br>(93%)", "Control Group", "Treatment Group"]
            values = [0.93, stats["control_churn"], stats["treatment_churn"]]
            colors = ["#95a5a6","#e74c3c","#2ecc71"]
            fig2 = go.Figure(go.Bar(
                x=labels, y=[v*100 for v in values],
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
            ))
            fig2.update_layout(
                title="SEBI Benchmark vs Intervention",
                yaxis_title="Rate (%)", yaxis_range=[0, 110],
                margin=dict(l=0,r=0,t=30,b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.success(f"""
        **Simulation Results:**  
        • Churn reduced by **{stats['churn_reduction']:.1%}** (from {stats['control_churn']:.1%} → {stats['treatment_churn']:.1%})  
        • User LTV extended **{ltv_mult}x** ({ltv_base}mo → {ltv_base*ltv_mult:.1f}mo)  
        • Avg trader PnL improvement: ₹{stats['trt_avg_pnl']-stats['ctrl_avg_pnl']:,.0f} per trader
        """)


def page_model_performance():
    st.title("📈 Model Performance")

    plots = {
        "elbow_silhouette.png":   "K-Means Elbow & Silhouette",
        "pca_clusters.png":       "PCA Cluster Visualization",
        "cohort_radar.png":       "Cohort Behavioral Radar",
        "risk_model_roc_fi.png":  "Risk Classifier – ROC & Feature Importance",
        "confusion_matrix.png":   "Risk Classifier – Confusion Matrix",
        "drawdown_regression.png":"Drawdown Regressor",
        "nudge_distribution.png": "Risk Nudge Distribution",
        "ab_simulation.png":      "A/B Simulation Results",
        "churn_waterfall.png":    "Churn Waterfall vs SEBI Benchmark",
    }

    available = {k:v for k,v in plots.items() if os.path.exists(os.path.join(PLOTS_DIR, k))}
    if not available:
        st.warning("No plots found. Run the pipeline scripts first.")
        st.code("""
cd /Users/gopichand/Desktop/TradeGuard
python data/generate_data.py
python analysis/segmentation.py
python analysis/risk_model.py
python simulation/ab_simulator.py
        """, language="bash")
        return

    for fname, title in available.items():
        st.subheader(title)
        path = os.path.join(PLOTS_DIR, fname)
        st.image(path, use_container_width=True)
        st.divider()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    page = sidebar()

    if not setup_check():
        st.warning("⚠️  Database / models not found. Running setup …")
        with st.spinner("Generating data & training models. This takes ~60s …"):
            import subprocess, sys
            root = ROOT
            subprocess.run([sys.executable, os.path.join(root,"data","generate_data.py")],  check=True)
            subprocess.run([sys.executable, os.path.join(root,"analysis","segmentation.py")], check=True)
            subprocess.run([sys.executable, os.path.join(root,"analysis","risk_model.py")],   check=True)
            subprocess.run([sys.executable, os.path.join(root,"simulation","ab_simulator.py")], check=True)
        st.success("Setup complete! Reloading …")
        st.rerun()

    orders, traders, summary = load_db()

    if   page == "📊 Dashboard Overview":
        page_overview(orders, traders, summary)
    elif page == "👥 Trader Cohort Explorer":
        page_cohort_explorer(summary, traders)
    elif page == "⚠️ Risk Nudge Monitor":
        page_risk_nudge(orders, traders)
    elif page == "🔴 Pre-Trade Kill-Switch":
        page_kill_switch()
    elif page == "🧪 A/B Intervention Simulator":
        page_ab_simulator()
    elif page == "📈 Model Performance":
        page_model_performance()


if __name__ == "__main__":
    main()

