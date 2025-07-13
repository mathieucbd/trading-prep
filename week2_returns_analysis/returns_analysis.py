import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = yf.Ticker("AAPL")
data = ticker.history(period="1y")
plt.plot(data)

# print(data.head())

daily_returns = data["Close"].pct_change().dropna()
monthly_returns = daily_returns.resample('ME').sum()
mean_return_daily = daily_returns.mean()
mean_return_ann = (1 + mean_return_daily) ** 252 - 1
volatility = daily_returns.std() * (252 ** 0.5)

print(f'ann return: {mean_return_ann:.2%}, vol: {volatility:.2%} ')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
returns_filepath = os.path.join(ROOT_DIR, 'returns.csv')  # requires `import os`

returns = {"daily_returns": daily_returns,
           "monthly_returns": monthly_returns}

df = pd.DataFrame(returns)
df.to_csv(returns_filepath)