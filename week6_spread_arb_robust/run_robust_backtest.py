from robust_spread_arb import PairBacktester
import pandas as pd
import json
import os

def run_backtest():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "config.json")

    # Define backtest parameters __> put json file
    with open(config_dir, "r") as params_file:
        params = json.load(params_file)

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