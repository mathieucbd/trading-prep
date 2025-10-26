# Trading Preparation

This repository documents my **ongoing journey to develop my skills** in quantitative trading, systematic strategies, and market data analysis. It is a **progressive and evolving project**, designed to guide a beginner in quant trading from Python basics to advanced statistical arbitrage and portfolio-level systems.

---

## 📅 Project Overview

This project follows a **weekly progression**, where each week builds on the previous one and gradually increases in difficulty. The plan starts with Python foundations and simple strategies, then moves toward more complex topics like machine learning alpha integration and live strategy simulation.

> ⚠️ The preparation is still ongoing, and the repository structure will continue to evolve. The description below may not reflect the exact latest state of the project.

---

## 📁 Current Repository Structure

```bash
mathieucbd/
│
├── week3_sma_strategy/          # SMA crossover backtest (long-only)
├── week4_rsi_strategy/          # RSI-based strategy backtest
├── week5_spread_arb/            # Simple 2-asset arbitrage model
├── week6_spread_arb_robust/     # Enhanced arbitrage with dynamic z-score logic
│
├── .gitignore                   # Ignore cache and system files
└── README.md                    # This file
```

> The early folders (`week1_intro/`, `week2_returns_analysis/`) are not uploaded because they were very simple foundational exercises (loops, lists, returns) and not particularly relevant to include here.

---

## 🧠 Learning Path

The roadmap is designed for **a beginner in quantitative trading** to gradually gain practical skills in:

- Python programming and data manipulation
- Strategy design and backtesting
- Statistical arbitrage and mean reversion logic
- Machine learning alpha integration
- Risk targeting and portfolio construction

Each week’s difficulty increases step-by-step, transitioning from simple indicators to multi-pair statistical arbitrage and live testing.

---

## ⚙️ Tools and Libraries

The project relies on common Python libraries for data handling, visualization, and financial backtesting. The list below is not exhaustive and may evolve as the project develops.

### Core dependencies:

```txt
numpy
pandas
matplotlib
yfinance
scikit-learn
ccxt
```

Other tools (like `pandas-datareader`, `requests`, and `peewee`) are occasionally used for data collection, local storage, and analysis.

---

## 🧩 Upcoming Work

- Portfolio-level statistical arbitrage engine
- Machine learning-based alpha filtering
- Volatility targeting and drawdown control
- Live crypto stat arb simulation using `ccxt`

---

## 📈 Objective

By the end of this project, the goals are to:

- Achieve proficiency in Python applied to trading
- Design and backtest multiple quantitative strategies
- Deploy a live statistical arbitrage strategy (likely on crypto for low transaction costs)
- Maintain a transparent and well-documented GitHub record of progress

