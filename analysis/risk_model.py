"""
TradeGuard – Drawdown Risk Model & Real-Time Risk Nudge Engine
Builds a gradient-boosted drawdown predictor and generates risk nudge signals.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, mean_absolute_error, r2_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

DB_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "tradeguard.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RISK_THRESHOLD = 5.0   # drawdown % above which we classify as HIGH-RISK


def load_order_features(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("""
        SELECT
            o.*,
            t.archetype,
            t.capital,
            t.experience_months,
            t.risk_tolerance
        FROM orders o
        JOIN traders t USING(trader_id)
    """, conn)
    conn.close()
    return df


def engineer_order_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_time"] = pd.to_datetime(df["order_time"])
    df["hour"]       = df["order_time"].dt.hour
    df["minute"]     = df["order_time"].dt.minute
    df["day_of_week"]= pd.to_datetime(df["trade_date"]).dt.dayofweek

    # Risk appetite proxy
    df["risk_score"] = df["risk_tolerance"].map({"low":0,"medium":1,"high":2}).fillna(1)

    # Consecutive-loss proxy (rolling PnL within trader)
    df = df.sort_values(["trader_id","order_time"])
    df["cum_pnl_trader"] = df.groupby("trader_id")["pnl"].cumsum()

    # Notional-to-capital ratio
    df["notional_pct_capital"] = (df["notional_value"] / df["capital"].clip(lower=1)) * 100

    # Binary risk label
    df["high_risk"] = (df["drawdown_pct"] > RISK_THRESHOLD).astype(int)

    return df


FEATURES = [
    "quantity", "leverage_ratio", "notional_pct_capital",
    "is_expiry_day", "revenge_flag", "overleveraged",
    "hour", "minute", "day_of_week",
    "experience_months", "risk_score",
]


def train_risk_classifier(df: pd.DataFrame):
    X = df[FEATURES].fillna(0)
    y = df["high_risk"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)
    auc    = roc_auc_score(y_te, y_prob)
    print(f"\n[Risk Classifier] AUC-ROC = {auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Low Risk","High Risk"]))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}", color="#e74c3c")
    ax1.plot([0,1],[0,1],"k--", lw=1)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Risk Classifier – ROC Curve"); ax1.legend()

    # Feature importance
    imp    = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    imp.plot(kind="barh", ax=ax2, color="#3498db")
    ax2.set_title("Feature Importance – GBM Risk Classifier")
    ax2.set_xlabel("Importance")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "risk_model_roc_fi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")

    # Confusion matrix
    cm  = confusion_matrix(y_te, y_pred)
    fig2, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Low","High"]).plot(ax=ax, colorbar=False)
    ax.set_title("Risk Classifier – Confusion Matrix")
    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()

    return clf, auc


def train_drawdown_regressor(df: pd.DataFrame):
    X = df[FEATURES].fillna(0)
    y = df["drawdown_pct"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    reg.fit(X_tr, y_tr)
    y_hat = reg.predict(X_te)
    mae   = mean_absolute_error(y_te, y_hat)
    r2    = r2_score(y_te, y_hat)
    print(f"\n[Drawdown Regressor] MAE={mae:.4f}  R²={r2:.4f}")

    # Actual vs predicted scatter (sample 2000)
    idx = np.random.choice(len(y_te), min(2000, len(y_te)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_te.iloc[idx], y_hat[idx], alpha=0.3, s=15, color="#9b59b6")
    lims = [min(y_te.min(), y_hat.min()), max(y_te.max(), y_hat.max())]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Drawdown %"); ax.set_ylabel("Predicted Drawdown %")
    ax.set_title(f"Drawdown Regressor  (MAE={mae:.4f}, R²={r2:.4f})")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "drawdown_regression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")

    return reg, mae, r2


def generate_risk_nudges(df: pd.DataFrame, clf) -> pd.DataFrame:
    """Assign real-time risk nudge level to every order."""
    X     = df[FEATURES].fillna(0)
    probs = clf.predict_proba(X)[:, 1]
    df    = df.copy()
    df["risk_prob"] = probs
    df["nudge_level"] = pd.cut(
        probs,
        bins=[0, 0.33, 0.66, 1.0],
        labels=["GREEN 🟢", "AMBER 🟡", "RED 🔴"],
    )
    return df


def plot_nudge_distribution(df: pd.DataFrame):
    counts = df["nudge_level"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ["#2ecc71", "#f39c12", "#e74c3c"]
    counts.sort_index().plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Real-Time Risk Nudge Distribution")
    ax.set_xlabel("Nudge Level"); ax.set_ylabel("Order Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "nudge_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")


def run_risk_modelling():
    print("Loading order features …")
    df = load_order_features(DB_PATH)
    print(f"  → {len(df):,} orders loaded")

    df = engineer_order_features(df)

    print("\nTraining risk classifier …")
    clf, auc = train_risk_classifier(df)

    print("\nTraining drawdown regressor …")
    reg, mae, r2 = train_drawdown_regressor(df)

    print("\nGenerating risk nudges …")
    df_nudged = generate_risk_nudges(df, clf)
    plot_nudge_distribution(df_nudged)

    # Save
    joblib.dump(clf, os.path.join(MODEL_DIR, "risk_classifier.pkl"))
    joblib.dump(reg, os.path.join(MODEL_DIR, "drawdown_regressor.pkl"))
    df_nudged.to_csv(os.path.join(MODEL_DIR, "orders_with_nudges.csv"), index=False)

    metrics = {"auc": auc, "mae": mae, "r2": r2}
    print(f"\n[✓] Risk modelling complete. Metrics: {metrics}")
    return clf, reg, df_nudged, metrics


if __name__ == "__main__":
    run_risk_modelling()

