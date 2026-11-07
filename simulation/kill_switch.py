"""
TradeGuard – Pre-Trade Kill-Switch Logic
Implements the pre-trade risk gate that blocks orders exceeding risk thresholds.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import joblib

DB_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "tradeguard.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURES = [
    "quantity", "leverage_ratio", "notional_pct_capital",
    "is_expiry_day", "revenge_flag", "overleveraged",
    "hour", "minute", "day_of_week",
    "experience_months", "risk_score",
]


class KillSwitch:
    """
    Pre-trade gate that blocks orders exceeding configurable risk thresholds.
    Rule hierarchy (any breach → BLOCK):
      1. Max leverage per trade
      2. Max drawdown % from intraday capital
      3. ML risk probability > threshold
      4. Revenge-trade rapid-fire (>N orders in T minutes)
      5. Expiry day over-leverage
    """

    def __init__(
        self,
        max_leverage:      float = 15.0,
        max_drawdown_pct:  float = 8.0,
        ml_prob_threshold: float = 0.70,
        revenge_window_min:int   = 10,
        revenge_max_orders:int   = 5,
    ):
        self.max_leverage       = max_leverage
        self.max_drawdown_pct   = max_drawdown_pct
        self.ml_prob_threshold  = ml_prob_threshold
        self.revenge_window_min = revenge_window_min
        self.revenge_max_orders = revenge_max_orders

        # Load pre-trained classifier if available
        clf_path = os.path.join(MODEL_DIR, "risk_classifier.pkl")
        self.clf = joblib.load(clf_path) if os.path.exists(clf_path) else None

    def evaluate(self, order: dict, trader: dict, recent_orders: list) -> dict:
        """
        Returns a dict:
            blocked   : bool
            reasons   : list[str]
            risk_prob : float (ML score)
            nudge     : str  (GREEN / AMBER / RED)
        """
        reasons   = []
        risk_prob = 0.0

        leverage = order.get("leverage_ratio", 0)
        if leverage > self.max_leverage:
            reasons.append(f"Leverage {leverage:.1f}x exceeds limit {self.max_leverage}x")

        capital  = trader.get("capital", 1)
        notional = order.get("notional_value", 0)
        dd_pct   = (notional / capital) * 100 if capital > 0 else 999
        if dd_pct > self.max_drawdown_pct * 10:          # rough proxy
            reasons.append(f"Notional {dd_pct:.1f}% of capital exceeds threshold")

        # Revenge trading check
        if len(recent_orders) >= self.revenge_max_orders:
            reasons.append(
                f"Rapid-fire: {len(recent_orders)} orders in last {self.revenge_window_min}m"
            )

        # ML risk score
        if self.clf is not None:
            feat = self._build_feature_row(order, trader)
            risk_prob = float(self.clf.predict_proba([feat])[0][1])
            if risk_prob > self.ml_prob_threshold:
                reasons.append(f"ML risk score {risk_prob:.2f} > threshold {self.ml_prob_threshold}")

        # Expiry day over-leverage
        if order.get("is_expiry_day") and leverage > 8:
            reasons.append(f"Expiry-day leverage {leverage:.1f}x > 8x hard-limit")

        blocked = len(reasons) > 0
        nudge   = ("RED 🔴"   if risk_prob > 0.66 else
                   "AMBER 🟡" if risk_prob > 0.33 else
                   "GREEN 🟢")

        return {
            "blocked":   blocked,
            "reasons":   reasons,
            "risk_prob": round(risk_prob, 4),
            "nudge":     nudge,
        }

    def _build_feature_row(self, order: dict, trader: dict) -> list:
        from datetime import datetime
        ot = pd.to_datetime(order.get("order_time", datetime.now()))
        risk_map = {"low": 0, "medium": 1, "high": 2}
        capital  = max(trader.get("capital", 1), 1)
        return [
            order.get("quantity", 0),
            order.get("leverage_ratio", 0),
            (order.get("notional_value", 0) / capital) * 100,
            int(order.get("is_expiry_day", 0)),
            int(order.get("revenge_flag", 0)),
            int(order.get("overleveraged", 0)),
            ot.hour, ot.minute, ot.dayofweek,
            trader.get("experience_months", 12),
            risk_map.get(trader.get("risk_tolerance","medium"), 1),
        ]


def demo_kill_switch():
    """Quick demo of the kill-switch on 20 random orders."""
    conn    = sqlite3.connect(DB_PATH)
    orders  = pd.read_sql("SELECT * FROM orders LIMIT 500", conn)
    traders = pd.read_sql("SELECT * FROM traders",          conn)
    conn.close()

    tmap = traders.set_index("trader_id").to_dict("index")
    ks   = KillSwitch()

    blocked_count = 0
    print(f"\n{'='*60}")
    print("Pre-Trade Kill-Switch Demo (first 20 high-leverage orders)")
    print(f"{'='*60}")

    shown = 0
    for _, row in orders.iterrows():
        result = ks.evaluate(row.to_dict(), tmap.get(row["trader_id"], {}), [])
        if result["blocked"]:
            blocked_count += 1
            if shown < 20:
                print(f"  ❌ BLOCKED  {row['order_id']}  |  {result['nudge']}  |  {result['reasons'][0]}")
                shown += 1

    total  = len(orders)
    pct    = blocked_count / total * 100
    print(f"\nResult: {blocked_count}/{total} orders blocked ({pct:.1f}%)\n")
    return blocked_count, total


if __name__ == "__main__":
    demo_kill_switch()

