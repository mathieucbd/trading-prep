import pandas as pd
import numpy as np

def calculate_sharpe(daily_returns, annualization_factor=252):
    daily_returns = daily_returns.dropna()
    volatility = daily_returns.std()

    if volatility == 0:
        return 0.0
    
    return ((daily_returns.mean() * annualization_factor) / (volatility * np.sqrt(annualization_factor)))

def calculate_sortino(daily_returns, annualization_factor=252):
    daily_returns = daily_returns.dropna()
    negative_returns = daily_returns[daily_returns < 0]
    downside_vol = negative_returns.std()

    if downside_vol == 0:
        return 0.0
    
    return ((daily_returns.mean() * annualization_factor) / (downside_vol * np.sqrt(annualization_factor)))
 
def calculate_hit_ratio(daily_returns):
    daily_returns = daily_returns.dropna()

    if len(daily_returns) == 0:
        return 0.0
    
    return (daily_returns[daily_returns > 0].count() / (daily_returns[daily_returns != 0].count()))

def calculate_max_drawdown(daily_returns):
    cum_pnl = 1 + daily_returns.cumsum()
    peak = cum_pnl.cummax()
    peak_safe = peak.replace(0, np.nan)
    drawdown = (cum_pnl - peak_safe) / peak_safe

    return drawdown.min()

def calculate_calmar(daily_returns, annualization_factor=252):
    daily_returns = daily_returns.dropna()

    if daily_returns.empty:
        return 0.0

    annual_return = daily_returns.mean() * annualization_factor
    max_dd = calculate_max_drawdown(daily_returns)
    
    if max_dd == 0 or np.isnan(max_dd):
        return 0.0

    return annual_return / abs(max_dd)
