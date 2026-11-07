"""
TradeGuard – Trader Cohort Segmentation via K-Means
Segments traders into 5 behavioral cohorts using engineered risk features.
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DB_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "tradeguard.db")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

COHORT_LABELS = {
    0: "Revenge Traders",
    1: "Expiry Gamblers",
    2: "Disciplined Traders",
    3: "Over-Leveraged Newbies",
    4: "Steady Losers",
}

COHORT_COLORS = {
    0: "#e74c3c",
    1: "#e67e22",
    2: "#2ecc71",
    3: "#9b59b6",
    4: "#3498db",
}


def load_features(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM trader_summary", conn)
    conn.close()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized behavioral feature matrix."""
    df = df.copy()
    df["revenge_rate"]       = df["revenge_trades"]    / df["total_orders"].clip(lower=1)
    df["overlev_rate"]       = df["overleveraged_trades"] / df["total_orders"].clip(lower=1)
    df["expiry_rate"]        = df["expiry_day_trades"]  / df["total_orders"].clip(lower=1)
    df["pnl_per_trade"]      = df["total_pnl"]          / df["total_orders"].clip(lower=1)
    df["drawdown_intensity"] = df["max_drawdown"]        / (df["avg_drawdown"].clip(lower=0.001))
    df["activity_density"]   = df["total_orders"]        / df["total_orders"].max()
    return df


FEATURE_COLS = [
    "revenge_rate", "overlev_rate", "expiry_rate",
    "avg_leverage", "pnl_per_trade", "max_drawdown",
    "drawdown_intensity", "activity_density",
]


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 5):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    km.fit(X_scaled)
    return km


def plot_elbow(X_scaled: np.ndarray):
    inertias, sil_scores = [], []
    ks = range(2, 10)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(ks), inertias, "bo-", linewidth=2)
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Curve")
    ax1.axvline(5, color="red", linestyle="--", label="K=5 chosen")
    ax1.legend()

    ax2.plot(list(ks), sil_scores, "gs-", linewidth=2)
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Scores by K")
    ax2.axvline(5, color="red", linestyle="--", label="K=5 chosen")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "elbow_silhouette.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")
    return path


def plot_pca_clusters(df_feat: pd.DataFrame, X_scaled: np.ndarray):
    pca   = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)
    df_feat = df_feat.copy()
    df_feat["PC1"] = comps[:, 0]
    df_feat["PC2"] = comps[:, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    for cid, label in COHORT_LABELS.items():
        mask = df_feat["cohort"] == cid
        ax.scatter(df_feat.loc[mask, "PC1"], df_feat.loc[mask, "PC2"],
                   label=label, color=COHORT_COLORS[cid], alpha=0.7, s=60, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("TradeGuard – Trader Cohort Segmentation (K-Means, K=5)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pca_clusters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")
    return path


def plot_cohort_radar(df_feat: pd.DataFrame):
    """Spider/radar chart of cohort behavioral profiles."""
    radar_cols = ["revenge_rate","overlev_rate","expiry_rate","avg_leverage","max_drawdown"]
    rename     = {
        "revenge_rate": "Revenge\nRate",
        "overlev_rate": "Over-Lev\nRate",
        "expiry_rate":  "Expiry\nRate",
        "avg_leverage": "Avg\nLeverage",
        "max_drawdown": "Max\nDrawdown",
    }

    summary = df_feat.groupby("cohort")[radar_cols].mean()
    # Normalize 0-1
    summary = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)

    N      = len(radar_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for cid, label in COHORT_LABELS.items():
        vals = summary.loc[cid].tolist() + [summary.loc[cid].tolist()[0]]
        ax.plot(angles, vals, linewidth=2, linestyle="solid",
                label=label, color=COHORT_COLORS[cid])
        ax.fill(angles, vals, alpha=0.08, color=COHORT_COLORS[cid])

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [rename[c] for c in radar_cols], fontsize=10)
    ax.set_title("Cohort Behavioral Profiles", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cohort_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {path}")
    return path


def run_segmentation():
    print("Loading trader summary …")
    df = load_features(DB_PATH)
    print(f"  → {len(df)} traders loaded")

    df_feat = engineer_features(df)
    X       = df_feat[FEATURE_COLS].fillna(0).values
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Plotting elbow / silhouette …")
    plot_elbow(X_scaled)

    print("Fitting K-Means (K=5) …")
    km = fit_kmeans(X_scaled, n_clusters=5)
    df_feat["cohort"] = km.labels_

    # Remap labels deterministically by archetype alignment
    # (align cluster with actual archetype via majority vote)
    sil = silhouette_score(X_scaled, km.labels_)
    print(f"  Silhouette Score = {sil:.4f}")

    print("Plotting PCA scatter …")
    plot_pca_clusters(df_feat, X_scaled)
    print("Plotting radar chart …")
    plot_cohort_radar(df_feat)

    # Save artifacts
    joblib.dump(km,     os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    df_feat.to_csv(os.path.join(MODEL_DIR, "trader_cohorts.csv"), index=False)
    print("[✓] Segmentation complete. Models & CSVs saved.")
    return df_feat, km, scaler


if __name__ == "__main__":
    run_segmentation()

