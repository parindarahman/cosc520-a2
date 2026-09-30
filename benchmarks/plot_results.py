"""Draws Figures 1-5 from results/random.csv.

    Figure 1 -- build time vs n          (log-log)
    Figure 2 -- mean search time vs n    (log-log)
    Figure 3 -- mean deletion time vs n  (log-log)
    Figure 4 -- memory vs n              (log-log)
    Figure 5 -- tree height vs n         (log x, linear y)

Each point is the mean over runs. Every tree keeps the same colour and
marker in all figures. Each figure is saved as a PNG in results/figures/.

Usage (from the project root, after running the benchmarks):
    python -m benchmarks.plot_results

"""

import os

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import pandas as pd  # noqa: E402

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
ORDER = ["BST", "AVL", "Red-Black", "Splay"]
# Red-Black uses hollow markers and a dash-dot line so it stays visible
# when its values coincide with AVL's (e.g. equal heights on random input).
STYLE = {
    "BST":       {"color": "#7f7f7f", "marker": "o"},
    "AVL":       {"color": "#1f77b4", "marker": "s"},
    "Red-Black": {"color": "#d62728", "marker": "^", "mfc": "none", "ls": "-."},
    "Splay":     {"color": "#2ca02c", "marker": "D"},
}

FIGURES = [
    # (file name, CSV column, y-axis label, log-scale y?)
    ("fig1_build_time", "build_s", "Build time (s)", True),
    ("fig2_search_time", "search_us", "Mean search time (\u00b5s)", True),
    ("fig3_delete_time", "delete_us", "Mean deletion time (\u00b5s)", True),
    ("fig4_memory", "memory_mb", "Memory (MB)", True),
    ("fig5_height", "height", "Tree height (nodes on longest path)", False),
]


def summarise(df, column):
    """
    Input: df -- results table; column -- name of a measured column.
    Output: DataFrame with columns n, structure, mean (averaged over runs).
    """
    return df.groupby(["n", "structure"])[column].mean().reset_index(name="mean")


def plot_figure(df, file_name, column, y_label, log_y, runs):
    """
    Input: df -- results table; file_name -- output name without extension;
           column -- CSV column to plot; y_label -- y-axis label;
           log_y -- use a log-scale y-axis; runs -- number of runs (for caption).
    Output: None (writes the figure as a PNG), or skips if no data.
    """
    summary = summarise(df.dropna(subset=[column]), column)
    if summary.empty:
        print(f"  skipped {file_name}: no data in column '{column}'")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for name in ORDER:
        rows = summary[summary["structure"] == name].sort_values("n")
        if rows.empty:
            continue
        ax.plot(rows["n"], rows["mean"], label=name, lw=1.5, ms=5, **STYLE[name])

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Number of nodes $n$")
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"mean of {runs} run{'s' if runs != 1 else ''}", fontsize=8,
                 loc="right", color="0.4")
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{file_name}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    """Loads results/random.csv and draws all five figures."""
    path = os.path.join(RESULTS_DIR, "random.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run 'python -m benchmarks.run_benchmarks' first.")
    df = pd.read_csv(path)
    runs = int(df["run"].max())
    for file_name, column, y_label, log_y in FIGURES:
        plot_figure(df, file_name, column, y_label, log_y, runs)


if __name__ == "__main__":
    main()