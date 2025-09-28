import json
import os
import logging
import argparse
from robust_spread_arb import PairBacktester
from plots import plot_pnl_curve, plot_spread_zscore, plot_drawdowns

def run_backtest():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "config.json")

    # --- CLI Parser ---
    parser = argparse.ArgumentParser(description="Run spread arbitrage backtest")
    parser.add_argument("--config", type=str, default=config_dir, help="Path to config.json")
    parser.add_argument("--window", type=float, help="Rolling window for z-score")
    parser.add_argument("--z_entry", type=float, help="Z-score entry threshold")
    parser.add_argument("--z_exit", type=float, help="Z-score exit threshold")
    parser.add_argument("--capital", type=float, help="Capital in USD")
    parser.add_argument("--hedge_ratio", type=float, help="Hedge ratio")
    args = parser.parse_args()

    # --- Load base config ---
    with open(config_dir, "r") as params_file:
        params = json.load(params_file)

    # --- Override config.json with CLI arguments if provided ---
    for keys in ["window", "z_entry", "z_exit", "capital", "hedge_ratio"]:
        val = getattr(args, keys)
        if val is not None:
            params[keys] = val

    # --- Logging setup ---
    os.makedirs(os.path.dirname(params["log_file"]), exist_ok=True)
    logging.basicConfig(
        filename=params["log_file"],
        level=logging.INFO, # Possible levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Starting backtets with parameters: %s", params)

    # --- Run Backtest ---
    bt = PairBacktester(**params)
    metrics = bt.run_backtest(verbose=True)

    # --- Export CSV ---
    os.makedirs(os.path.dirname(params["export_filename"]), exist_ok=True)
    bt.export_timeseries(filename=params["export_filename"])

    # --- Plots (delegated to plots.py) ---
    if params.get("plots", True):
        out1 = plot_pnl_curve(bt.pnl, out_dir="outputs", filename="pnl_curve.png")
        out2 = plot_spread_zscore(bt.spread, bt.z_score, out_dir="outputs", filename="spread_zscore.png")
        out3 = plot_drawdowns(bt.pnl, out_dir="outputs", filename="drawdowns.png")
        logging.info("Saved plots: %s, %s, %s", out1, out2, out3)

    logging.info("Backtest complete. Metrics: %s", metrics)
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        try:
            print(f"{k}: {v:.2f}")
        except Exception:
            print(f"{k}: {v}")

if __name__ == "__main__":
    run_backtest()