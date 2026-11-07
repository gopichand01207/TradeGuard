#!/usr/bin/env python3
"""
TradeGuard – Full Pipeline Runner
Runs all steps in sequence: data generation → segmentation → risk modelling → AB simulation
"""
import subprocess, sys, os, time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Data Generation",        os.path.join(ROOT, "data",       "generate_data.py")),
    ("Trader Segmentation",    os.path.join(ROOT, "analysis",   "segmentation.py")),
    ("Risk Modelling",         os.path.join(ROOT, "analysis",   "risk_model.py")),
    ("A/B Simulation",         os.path.join(ROOT, "simulation", "ab_simulator.py")),
]

def run():
    print("=" * 60)
    print("  TradeGuard – Full Pipeline")
    print("=" * 60)
    for i, (name, script) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {name} …")
        t0 = time.time()
        result = subprocess.run([sys.executable, script], capture_output=False)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  ❌ FAILED (exit {result.returncode})")
            sys.exit(result.returncode)
        print(f"  ✅ Done in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("  ✅ Pipeline complete!")
    print("  Run the dashboard:")
    print(f"     streamlit run {os.path.join(ROOT, 'app', 'dashboard.py')}")
    print("=" * 60)

if __name__ == "__main__":
    run()

