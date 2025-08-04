from backtest_spread_arb import PairBacktester
import pandas as pd

def main():
    # Define backtest parameters
    params = {
        "file1": "SPY.csv",
        "file2": "IVV.csv",
        "hedge_ratio": 0.992, # Determined empirically in spread_ar_concepts.ipynb
        "capital": 50_000_000,
        "transaction_costs": 0.0001, # 1bp
        "window": 20, # Determined empirically in spread_ar_concepts.ipynb
        "z_entry": 1.64, # 90% confidence level
        "z_exit": 1.28, # 80% confidence level
        "positioning_method": "net",
        "margin_rate": None,
    }

    # Instantiate and run backtest
    bt = PairBacktester(**params, data_subdir="data")
    metrics = bt.run_backtest(verbose=True)

    # Export time series to analyze data if needed
    # bt.export_timeseries("spread_arb_timeseries.csv")


    # Print results
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

if __name__ == "__main__":
    main()