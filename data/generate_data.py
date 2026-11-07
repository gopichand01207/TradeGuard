"""
TradeGuard - Data Generator
Generates synthetic intraday F&O order logs for 75,000+ records
simulating realistic retail trader behavior on NSE.
"""

import sqlite3
import pandas as pd
import numpy as np
from faker import Faker
import random
import os
from datetime import datetime, timedelta, date

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

# ── constants ──────────────────────────────────────────────────────────────────
N_TRADERS   = 500
N_ORDERS    = 76000          # >75,000 as per resume bullet
DB_PATH     = os.path.join(os.path.dirname(__file__), "tradeguard.db")
CSV_PATH    = os.path.join(os.path.dirname(__file__), "orders.csv")

SYMBOLS     = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]
SEGMENTS    = ["FO"]         # Futures & Options only
EXPIRY_DAYS = [3, 10, 17, 24, 31]   # typical weekly/monthly expiry dates

# Trader cohort archetypes (used to seed behavioral patterns)
ARCHETYPES = {
    "revenge_trader":       0.22,   # 22% of base pool
    "expiry_gambler":       0.18,
    "disciplined_trader":   0.15,
    "overlevered_newbie":   0.25,
    "steady_loser":         0.20,
}

def generate_traders(n: int) -> pd.DataFrame:
    rows = []
    archetype_pool = []
    for arch, frac in ARCHETYPES.items():
        archetype_pool.extend([arch] * int(n * frac))
    # pad to exactly n
    while len(archetype_pool) < n:
        archetype_pool.append("steady_loser")
    random.shuffle(archetype_pool)

    for i in range(n):
        arch = archetype_pool[i]
        # Base account parameters depend on archetype
        capital = {
            "revenge_trader":     np.random.randint(80_000, 500_000),
            "expiry_gambler":     np.random.randint(50_000, 300_000),
            "disciplined_trader": np.random.randint(200_000, 2_000_000),
            "overlevered_newbie": np.random.randint(30_000, 150_000),
            "steady_loser":       np.random.randint(100_000, 800_000),
        }[arch]

        rows.append({
            "trader_id":   f"TRD{i+1:05d}",
            "name":        fake.name(),
            "city":        fake.city(),
            "archetype":   arch,
            "capital":     capital,
            "experience_months": np.random.randint(1, 120),
            "risk_tolerance": np.random.choice(["low","medium","high"],
                              p=[0.3, 0.4, 0.3]),
            "joined_date": fake.date_between(start_date="-5y", end_date="-3m"),
        })
    return pd.DataFrame(rows)


def _make_order(trader_id, arch, capital, trade_date, seq, symbols):
    """Generate a single synthetic order row."""
    symbol   = random.choice(symbols)
    side     = random.choice(["BUY","SELL"])
    is_expiry = trade_date.day in EXPIRY_DAYS

    # Archetype-driven lot multipliers
    base_lots = {
        "revenge_trader":     np.random.randint(5, 30),
        "expiry_gambler":     np.random.randint(10, 50) if is_expiry else np.random.randint(1, 10),
        "disciplined_trader": np.random.randint(1, 8),
        "overlevered_newbie": np.random.randint(8, 40),
        "steady_loser":       np.random.randint(2, 15),
    }[arch]

    lot_size  = {"NIFTY":25,"BANKNIFTY":15,"FINNIFTY":40,"MIDCPNIFTY":75,"SENSEX":10}[symbol]
    qty       = base_lots * lot_size
    price     = round(np.random.uniform(18000, 50000) if "NIFTY" in symbol else
                      np.random.uniform(100, 500), 2)

    # Simulate PnL per order
    pnl_pct   = np.random.normal(-0.003, 0.025)   # slight negative drift (retail reality)
    pnl       = round(price * qty * pnl_pct, 2)

    # Drawdown as % of capital — capped at 100%
    drawdown  = round(min(-min(pnl, 0) / capital * 100, 100.0), 4) if capital > 0 else 0

    # Revenge trading flag: rapid succession of orders after a loss
    revenge   = int(arch == "revenge_trader" and seq > 1 and np.random.random() < 0.45)

    # Over-leveraged flag
    notional  = price * qty
    leverage  = round(notional / capital, 2)
    overleveraged = int(leverage > 10)

    order_time = datetime.combine(trade_date, datetime.min.time()) + \
                 timedelta(hours=np.random.randint(9, 15),
                           minutes=np.random.randint(0, 59),
                           seconds=np.random.randint(0, 59))

    return {
        "order_id":        f"ORD{trader_id}{trade_date.strftime('%Y%m%d')}{seq:04d}",
        "trader_id":       trader_id,
        "symbol":          symbol,
        "segment":         "FO",
        "side":            side,
        "quantity":        qty,
        "price":           price,
        "order_time":      order_time.strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date":      trade_date.strftime("%Y-%m-%d"),
        "pnl":             pnl,
        "drawdown_pct":    drawdown,
        "is_expiry_day":   int(is_expiry),
        "revenge_flag":    revenge,
        "overleveraged":   overleveraged,
        "leverage_ratio":  leverage,
        "notional_value":  round(notional, 2),
    }


def generate_orders(traders: pd.DataFrame, n_orders: int) -> pd.DataFrame:
    rows      = []
    dates_pool = pd.date_range("2023-01-01", "2024-10-31", freq="B").tolist()
    dates_pool = [d.date() for d in dates_pool]

    trader_list = traders.to_dict("records")
    per_trader  = n_orders // len(trader_list)
    remainder   = n_orders % len(trader_list)

    for idx, t in enumerate(trader_list):
        n_t = per_trader + (1 if idx < remainder else 0)
        # pick random trade dates for this trader
        t_dates = [random.choice(dates_pool) for _ in range(n_t)]
        t_dates.sort()
        for seq, td in enumerate(t_dates):
            rows.append(_make_order(
                t["trader_id"], t["archetype"],
                t["capital"], td, seq+1, SYMBOLS
            ))

    return pd.DataFrame(rows)


def save_to_sqlite(traders: pd.DataFrame, orders: pd.DataFrame, db_path: str):
    conn = sqlite3.connect(db_path)
    traders.to_sql("traders", conn, if_exists="replace", index=False)
    orders.to_sql("orders",  conn, if_exists="replace", index=False)
    # Create useful views
    conn.execute("""
        CREATE VIEW IF NOT EXISTS trader_summary AS
        SELECT
            o.trader_id,
            t.archetype,
            t.capital,
            COUNT(*)                        AS total_orders,
            SUM(o.pnl)                      AS total_pnl,
            AVG(o.drawdown_pct)             AS avg_drawdown,
            MAX(o.drawdown_pct)             AS max_drawdown,
            SUM(o.revenge_flag)             AS revenge_trades,
            SUM(o.overleveraged)            AS overleveraged_trades,
            SUM(o.is_expiry_day)            AS expiry_day_trades,
            AVG(o.leverage_ratio)           AS avg_leverage
        FROM orders o
        JOIN traders t USING(trader_id)
        GROUP BY o.trader_id
    """)
    conn.commit()
    conn.close()
    print(f"[✓] SQLite DB saved → {db_path}")


def main():
    print("Generating traders …")
    traders = generate_traders(N_TRADERS)
    print(f"  → {len(traders)} traders")

    print("Generating orders …")
    orders  = generate_orders(traders, N_ORDERS)
    print(f"  → {len(orders)} orders")

    # Save
    save_to_sqlite(traders, orders, DB_PATH)
    orders.to_csv(CSV_PATH, index=False)
    print(f"[✓] CSV saved         → {CSV_PATH}")
    return traders, orders


if __name__ == "__main__":
    main()
