from calendar import c
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def moving_average(sequence: np.ndarray, window_size: int) -> np.ndarray:
    """
    Simple moving average. For length N and window k, returns length N-k+1.
    """
    if window_size is None or window_size <= 1:
        return sequence
    if window_size > sequence.size:
        return sequence
    cumsum = np.cumsum(np.insert(sequence, 0, 0.0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def load_series(path_str: str) -> np.ndarray:
    """
    Load a 1D numeric series from .npy or .csv (comma or whitespace separated).
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr).squeeze()
        if arr.ndim != 1:
            raise ValueError(".npy must contain a 1D array")
        return arr.astype(float)
    # Fallback: CSV or txt
    try:
        # Try comma-separated first
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        # Fallback to any whitespace
        arr = np.loadtxt(path)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError("CSV/TXT must contain a single 1D series")
    return arr.astype(float)


def plot_two_series(series1: np.ndarray,
                    series2: np.ndarray,
                    label1: str,
                    label2: str,
                    title: str,
                    save_path: Path) -> None:
    """
    Plot two line charts with fill between each curve and y=0, and '*' markers.
    """
    x = np.arange(series1.size)
    fig, ax = plt.subplots()
    c1, c11 = '#8dee5c', '#97d65b'
    c2, c22 = '#e45a41', '#e67860'

    # First series: line + '*' markers + fill to y=0
    ax.plot(x, series1, color=c1, marker='*', markersize=6, linewidth=1.5, label=label1)
    ax.fill_between(x, series1, 0, color=c11, alpha=0.2)

    # Second series: line + '*' markers + fill to y=0
    ax.plot(x, series2, color=c2, marker='*', markersize=6, linewidth=1.5, label=label2)
    ax.fill_between(x, series2, 0, color=c22, alpha=0.2)

    # Baseline y=0 for reference
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.3)

    if title:
        ax.set_title(title)
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)


def main(smooth, title):
    save = 'checkpoint/fig/ablationC.png'
    s1 = np.random.rand(100) +1  # 随机生成一个长度为100的array向量
    s2 = np.random.rand(100)

    if smooth> 1:
        s1 = moving_average(s1, smooth)
        s2 = moving_average(s2, smooth)

    # Align lengths by trimming to the shorter one
    min_len = min(s1.size, s2.size)
    s1 = s1[:min_len]
    s2 = s2[:min_len]

    plot_two_series(s1, s2, 'line1', 'line2', title, Path(save))
    print(f"Saved figure to: {save}")


if __name__ == "__main__":
    main(1, 'test')


