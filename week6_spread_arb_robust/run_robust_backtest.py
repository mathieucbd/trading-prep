import json
import os
import logging
import argparse
from datetime import datetime
from robust_spread_arb import PairBacktester
import matplotlib as plt

def run_backtest():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "config.json")

    # --- CLI Parser ---
    parser = argparse.ArgumentParser(description="Run spread arbitrage backtest")
    parser.add_argument("--config", type=str, default=config_dir, help="Path to config.json")
    parser.add_argument("--window", type=float, help="Rolling window for z-score")
    parser.add_argument("--z_entry", type=float, help="Z_score entry threshold")
    parser.add_argument("--z_exit", type=float, help="Z_score exit threshold")
    parser.add_argument("--capital", type=float, help="Capital in USD")
    parser.add_argument("--plots", type=float, default=True, help="Display plots")
    args = parser.parse_args()

    # --- Load base config ---
    with open(config_dir, "r") as params_file:
        params = json.load(params_file)

    # --- Override config.json with CLI arguments if provided ---
    for keys in ["window", "z_entry", "z_exit", "capital", "plots"]:
        val = getattr(args, keys)
        if val is not None:
            params[keys] = val

    # Instantiate and run backtest
    bt = PairBacktester(**params)
    metrics = bt.run_backtest(verbose=True)

    # Export time series to analyze data if needed
    # bt.export_timeseries("spread_arb_timeseries.csv")


    # Print results
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    run_backtest()