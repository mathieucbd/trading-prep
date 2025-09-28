from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_pnl_curve(pnl, out_dir="outputs", filename="pnl_curve.png") -> str:
    out = _ensure_dir(out_dir) / filename
    equity = pnl.cumsum()
    plt.figure(figsize=(10, 5))
    equity.plot()
    plt.title("Cumulative PnL")
    plt.ylabel("PnL ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return str(out)

def plot_spread_zscore(spread, z_score, out_dir="outputs", filename="spread_zscore.png") -> str:
    out = _ensure_dir(out_dir) / filename
    # scale z-score to spread units for overlay (mean/std on overlapping sample)
    s_mean = spread.mean()
    s_std = spread.std()
    z_scaled = z_score * s_std + s_mean

    plt.figure(figsize=(10, 5))
    spread.plot(label="Spread")
    plt.plot(z_scaled, label="Z-score (scaled)")
    plt.title("Spread and Z-score (overlay)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return str(out)

def plot_drawdowns(pnl, out_dir="outputs", filename="drawdowns.png") -> str:
    out = _ensure_dir(out_dir) / filename
    equity = pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    # avoid divide by zero at the start
    peak_safe = peak.replace(0, np.nan)
    drawdown_pct = drawdown / peak_safe

    plt.figure(figsize=(10, 4))
    drawdown_pct.plot()
    plt.title("Drawdown (%)")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return str(out)
