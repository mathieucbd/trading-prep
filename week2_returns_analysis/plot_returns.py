import returns_analysis as src
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


df = pd.DataFrame({"daily_returns":src.daily_returns,
                   "monthly_returns": src.monthly_returns})

plt.plot(df.index, df["daily_returns"])
plt.plot(df.index, df["monthly_returns"])
plt.xlabel("Time")
plt.ylabel("Returns")
plt.title("Daily and monthly returns of ME")
plt.show()