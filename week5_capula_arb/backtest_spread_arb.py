import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_sharpe, calculate_sortino, calculate_hit_ratio, calculate_max_drawdown, calculate_calmar


class PairBacktester:
    def __init__(self, file1, file2, data_subdir="data", hedge_ratio=1, transaction_costs=0.0001, window=100, z_entry=2.0, z_exit=0.5, capital=50_000_000, positioning_method="net", margin_rate=0.1):
        # Resolve absolute path to the data directory relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(script_dir, data_subdir)
        
        # Build paths
        self.file1 = os.path.join(self.data_dir, file1)
        self.file2 = os.path.join(self.data_dir, file2)

        # Extract asset labels from filenames
        self.label1 = os.path.splitext(file1)[0].upper()
        self.label2 = os.path.splitext(file2)[0].upper()

        self.hedge_ratio = hedge_ratio
        self.transaction_costs = transaction_costs
        self.window=window
        self.z_entry=z_entry
        self.z_exit=z_exit

        # Internal containers
        self.data = None
        self.spread = None
        self.z_score = None
        self.position = None
        self.units = None
        self.unit_cost = None
        self.trade_entry = None
        self.cost = None
        self.pnl = None
        self.daily_returns = None

        self.capital = capital
        self.positioning_method = positioning_method
        self.margin_rate = margin_rate

    
    def load_data(self):
        df1 = pd.read_csv(self.file1, index_col=0, parse_dates=True)
        df2 = pd.read_csv(self.file2, index_col=0, parse_dates=True)
        self.data = pd.DataFrame({
            self.label1: df1["Close"] if "Close" in df1.columns else df1.iloc[:, 0],
            self.label2: df2["Close"] if "Close" in df2.columns else df2.iloc[:, 0],
        }).dropna()


    def compute_spread(self):
        price1 = self.data[self.label1]
        price2 = self.data[self.label2]

        self.spread = price1 - self.hedge_ratio * price2
        
        rolling_mean = self.spread.rolling(window=self.window).mean()
        rolling_std = self.spread.rolling(window=self.window).std()

        self.z_score = (self.spread - rolling_mean) / rolling_std
    

    def generate_signals(self):
        z_score_lagged = self.z_score.shift(1)

        entry_long = z_score_lagged < -self.z_entry
        entry_short = z_score_lagged > self.z_entry
        exit = (z_score_lagged > -self.z_exit) & (z_score_lagged < self.z_exit)

        position = []
        current_pos = 0

        for long_sig, short_sig, exit_sig in zip(entry_long, entry_short, exit):
            if exit_sig:
                current_pos = 0
            elif long_sig:
                current_pos = 1
            elif short_sig:
                current_pos = -1
            position.append(current_pos)
        
        self.position = pd.Series(position, index=self.data.index)
    

    def compute_pnl(self):
        price1 = self.data[self.label1]
        price2 = self.data[self.label2]

        # Per-unit cost of a spread (depending on the positioning method)
        if self.positioning_method == "gross":
            unit_cost = price1 + (self.hedge_ratio * price2).abs()
        elif self.positioning_method == "net":
            unit_cost = np.maximum(price1, (self.hedge_ratio * price2).abs())
        elif self.positioning_method == "margin":
            unit_cost = (price1 + (self.hedge_ratio * price2).abs()) * self.margin_rate
        else:
            raise ValueError("Invalid positioning_method. Use 'gross', 'net' or 'margin'.")
        
        unit_cost = unit_cost.replace(0, np.nan)
        self.unit_cost = unit_cost
        units = self.capital / unit_cost
        self.units = units

        spread_return = self.spread.diff()
        position_lagged = self.position.shift(1).fillna(0) # Shift position to avoid lookahead bias (necessary additionally to the shift in z_score_lagged)

        raw_pnl = position_lagged * units * spread_return

        # Transaction cost when a trade is done (when position change)
        trade_entry = (self.position != 0) & (position_lagged != self.position)
        self.trade_entry = trade_entry
        cost = trade_entry.shift(1).fillna(False).astype(float) * self.transaction_costs * self.spread.abs() * units
        self.cost = cost
        
        self.pnl = raw_pnl - cost
     
    def compute_metrics(self):
        if self.pnl is None:
            raise ValueError("PnL not computed yet. Run compute_pnl() first.")
        
        daily_returns = self.pnl / self.capital
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
        self.daily_returns = daily_returns

        metrics = {
            "Annualize Return (%)": 100 * self.daily_returns.mean() * 252,
            "Volatility (%)": 100 * daily_returns.std() * np.sqrt(252),
            "Sharpe Ratio": calculate_sharpe(self.daily_returns),
            "Sortino Ratio": calculate_sortino(self.daily_returns),
            "Max Drawdown (%)": 100 * calculate_max_drawdown(self.daily_returns),
            "Calmar Ratio": calculate_calmar(self.daily_returns),
            "Hit Ratio (%)": 100 * calculate_hit_ratio(self.daily_returns),
        }
        return metrics
    
    def run_backtest(self, verbose=True, compute_metrics=True):
        self.load_data()
        self.compute_spread()
        self.generate_signals()
        self.compute_pnl()

        if verbose:
            print("✅ Backtest complete")
            print(f"Final cumulative PnL: ${self.pnl.cumsum().iloc[-1]:.2f}")
            print(f"Final cumulative return: {(self.pnl.cumsum().iloc[-1] / self.capital):.2%}")
            print(f"Total trades: {self.trade_entry.sum()}")
        
        if compute_metrics:
            return self.compute_metrics()
        
    
    def export_timeseries(self, filename="backtest_timeseries.csv"):
        if self.data is None or self.spread is None or self.pnl is None:
            raise ValueError("Backtest not complete. Run run_backtest() first.")

        # Get current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filename)

        price1 = self.data[self.label1]
        price2 = self.data[self.label2]

        df_export = pd.DataFrame({
            f"{self.label1}_price": price1,
            f"{self.label2}_price": price2,
            "spread": self.spread,
            "z_score": self.z_score,
            "position": self.position,
            "units": self.units,
            "transaction_cost": self.cost,
            "pnl": self.pnl,
            "daily_returns": self.daily_returns,
            "trade_entry": self.trade_entry.astype(float),
        })

        df_export.to_csv(full_path)
        print(f"✅ Time series exported to {full_path}")



