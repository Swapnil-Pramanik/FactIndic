"""Column-width figure for the paper: the four-prompt comparison (Fig. 2), redrawn tall so it
stays readable at 3.5 in. Reads results/prompting/prompting_metrics.csv -> paper/fig_prompting.png.

Usage (from the project root): .venv/bin/python paper/figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

OUT = Path(__file__).parent / "fig_prompting.png"
MEASURES = [("recall", "Fact recall"), ("unsupported_per_summary", "Unsupported / summary"),
            ("nli_contradicted", "NLI-contradicted share")]
TITLES = {"1b": "Llama 3.2 1B", "gemma9b": "Gemma 2 9B"}


def main():
    table = pd.read_csv(config.PROMPTING_METRICS_PATH)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 8})
    fig, axes = plt.subplots(len(MEASURES), 2, figsize=(3.5, 4.4), sharex=True, sharey="row")
    for c, tag in enumerate(config.PROMPTING_MODELS):
        for r, (m, label) in enumerate(MEASURES):
            ax = axes[r, c]
            for prompt in config.ALL_PROMPTS:
                part = table[(table["model"] == tag) & (table["prompt"] == prompt)]
                ax.plot(part["level"], part[m], marker="o", markersize=3, linewidth=1.2,
                        color=config.PROMPT_COLORS_ALL[prompt], label=prompt)
            ax.set_xticks(config.SCALE_LEVELS)
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(TITLES[tag], fontsize=8.5)
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5)
            if r == len(MEASURES) - 1:
                ax.set_xlabel("mixing level (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6, w_pad=0.6)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
