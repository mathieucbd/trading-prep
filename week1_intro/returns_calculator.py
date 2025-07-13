import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def get_returns(ticker_symbol, period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    returns = data['Close'].pct_change().dropna()
    return returns

def plot_returns(ticker, period="1y"):
    data = yf.Ticker(ticker).history(period=period)
    returns = data["Close"].pct_change().dropna()

    returns.plot(title=f"Daily Returns: {ticker}", figsize=(10, 4))
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    returns = get_returns("AAPL")
    print(returns.agg(["count", "std", "mean", "min", "max", "skew", "kurt"]))

    risk_return = ((1 + returns.mean())**252 - 1) / (returns.std()*(252**0.5))
    print(f'risk_return: {risk_return}')

    plot_returns("AAPL")
