# analysis.py — Summary statistics and plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Statistics ────────────────────────────────────────────────────────────

def summary_stats(results_df):
    """
    Return a DataFrame summarising key metrics across simulations.

    Columns: age, median and percentiles for w1, w2, consumption, taxes, A1, A2.
    """
    metrics = ["w1", "w2", "w3", "consumption", "taxes", "A1", "A2", "A3"]
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    rows = []
    for age, grp in results_df.groupby("age"):
        row = {"age": age}
        for col in metrics:
            for q in quantiles:
                label = f"{col}_p{int(q*100)}"
                row[label] = grp[col].quantile(q)
            row[f"{col}_mean"] = grp[col].mean()
        rows.append(row)

    return pd.DataFrame(rows).set_index("age")


def lifetime_tax_summary(results_df):
    """Return per-simulation total lifetime taxes as a Series."""
    return results_df.groupby("sim_id")["taxes"].sum()


# ── Plots ─────────────────────────────────────────────────────────────────

def _dollar_fmt(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1_000:.0f}k")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def plot_withdrawal_paths(results_df, title=None, save_path=None):
    """
    Plot median ± 10th–90th percentile band for w1, w2, and w3 vs age.
    """
    stats = summary_stats(results_df)
    ages = stats.index

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(title or "Optimal Withdrawal Paths", fontsize=13)

    for ax, col, label, color in [
        (axes[0], "w1", "Cash (muni) withdrawals", "steelblue"),
        (axes[1], "w2", "Traditional IRA withdrawals", "darkorange"),
        (axes[2], "w3", "Roth IRA withdrawals", "mediumseagreen"),
    ]:
        ax.fill_between(ages, stats[f"{col}_p10"], stats[f"{col}_p90"],
                        alpha=0.2, color=color, label="10th–90th pct")
        ax.fill_between(ages, stats[f"{col}_p25"], stats[f"{col}_p75"],
                        alpha=0.35, color=color, label="25th–75th pct")
        ax.plot(ages, stats[f"{col}_p50"], color=color, lw=2, label="Median")
        ax.set_title(label)
        ax.set_xlabel("Age")
        ax.legend(fontsize=8)
        _dollar_fmt(ax)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_account_balances(results_df, title=None, save_path=None):
    """
    Plot median ± percentile band for A1, A2, and A3 account balances vs age.
    """
    stats = summary_stats(results_df)
    ages = stats.index

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(title or "Account Balance Paths", fontsize=13)

    for ax, col, label, color in [
        (axes[0], "A1", "Cash (muni) account balance", "steelblue"),
        (axes[1], "A2", "Traditional IRA balance", "darkorange"),
        (axes[2], "A3", "Roth IRA balance", "mediumseagreen"),
    ]:
        ax.fill_between(ages, stats[f"{col}_p10"], stats[f"{col}_p90"],
                        alpha=0.2, color=color, label="10th–90th pct")
        ax.fill_between(ages, stats[f"{col}_p25"], stats[f"{col}_p75"],
                        alpha=0.35, color=color, label="25th–75th pct")
        ax.plot(ages, stats[f"{col}_p50"], color=color, lw=2, label="Median")
        ax.set_title(label)
        ax.set_xlabel("Age")
        ax.legend(fontsize=8)
        _dollar_fmt(ax)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_tax_distribution(results_df, title=None, save_path=None):
    """
    Histogram of total lifetime taxes across simulations.
    """
    lifetime = lifetime_tax_summary(results_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lifetime / 1_000, bins=50, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.axvline(lifetime.median() / 1_000, color="black", lw=1.5, linestyle="--",
               label=f"Median: ${lifetime.median()/1_000:.0f}k")
    ax.set_xlabel("Total lifetime taxes ($000s)")
    ax.set_ylabel("Simulations")
    ax.set_title(title or "Distribution of Total Lifetime Taxes")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_consumption(results_df, title=None, save_path=None):
    """
    Three-panel consumption chart.

    Left:   Optimal strategy — 10/90 and 25/75 shaded bands + median, with
            dotted median lines for CRT and TRC benchmarks overlaid.
    Middle: CRT benchmark (Cash→Roth→Traditional) — full shaded bands + median.
    Right:  TRC benchmark (Traditional→Roth→Cash) — full shaded bands + median.
    """
    stats = summary_stats(results_df)
    ages  = stats.index

    has_crt = "consumption_crt" in results_df.columns
    has_trc = "consumption_trc" in results_df.columns

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(title or "After-Tax Annual Consumption", fontsize=13)

    # ── Left: Optimal ─────────────────────────────────────────────────────
    ax = axes[0]
    color_opt = "forestgreen"
    ax.fill_between(ages, stats["consumption_p10"], stats["consumption_p90"],
                    alpha=0.20, color=color_opt, label="10th–90th pct")
    ax.fill_between(ages, stats["consumption_p25"], stats["consumption_p75"],
                    alpha=0.35, color=color_opt, label="25th–75th pct")
    ax.plot(ages, stats["consumption_p50"], color=color_opt, lw=2, label="Median")

    if has_crt:
        p50_crt = results_df.groupby("age")["consumption_crt"].median()
        ax.plot(ages, p50_crt, color="steelblue", lw=1.5, linestyle="--",
                label="CRT median")
    if has_trc:
        p50_trc = results_df.groupby("age")["consumption_trc"].median()
        ax.plot(ages, p50_trc, color="darkorange", lw=1.5, linestyle="--",
                label="TRC median")

    ax.set_title("Optimal")
    ax.set_xlabel("Age")
    ax.legend(fontsize=8)
    _dollar_fmt(ax)
    ax.grid(alpha=0.3)

    # ── Middle: CRT benchmark ─────────────────────────────────────────────
    ax = axes[1]
    if has_crt:
        grp = results_df.groupby("age")["consumption_crt"]
        ax.fill_between(ages, grp.quantile(0.10), grp.quantile(0.90),
                        alpha=0.20, color="steelblue", label="10th–90th pct")
        ax.fill_between(ages, grp.quantile(0.25), grp.quantile(0.75),
                        alpha=0.35, color="steelblue", label="25th–75th pct")
        ax.plot(ages, grp.quantile(0.50), color="steelblue", lw=2, label="Median")
    ax.set_title("CRT Benchmark (Cash→Roth→Trad)")
    ax.set_xlabel("Age")
    ax.legend(fontsize=8)
    _dollar_fmt(ax)
    ax.grid(alpha=0.3)

    # ── Right: TRC benchmark ──────────────────────────────────────────────
    ax = axes[2]
    if has_trc:
        grp = results_df.groupby("age")["consumption_trc"]
        ax.fill_between(ages, grp.quantile(0.10), grp.quantile(0.90),
                        alpha=0.20, color="darkorange", label="10th–90th pct")
        ax.fill_between(ages, grp.quantile(0.25), grp.quantile(0.75),
                        alpha=0.35, color="darkorange", label="25th–75th pct")
        ax.plot(ages, grp.quantile(0.50), color="darkorange", lw=2, label="Median")
    ax.set_title("TRC Benchmark (Trad→Roth→Cash)")
    ax.set_xlabel("Age")
    ax.legend(fontsize=8)
    _dollar_fmt(ax)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_consumption_quantiles(results_df, title=None, save_path=None):
    """
    Three-panel quantile (inverse CDF) plot comparing annual consumption
    distributions across Optimal, CRT, and TRC strategies.

    X-axis: percentile (5th–95th). Y-axis: annual consumption ($), shared
    across panels. A dashed median reference line is drawn on each panel.
    """
    opt = results_df.groupby("sim_id")["consumption"].mean()
    crt = results_df.groupby("sim_id")["consumption_crt"].mean()
    trc = results_df.groupby("sim_id")["consumption_trc"].mean()

    # pcts descending: 95→5; prob = 100-pcts ascending: 5→95
    # q_vals descend (high→low consumption), giving a downward-sloping curve
    pcts = np.linspace(95, 5, 500)
    prob = 100 - pcts   # probability of sustaining at least q_vals[i], 5%→95%

    panels = [
        (opt, "Optimal",               "forestgreen"),
        (crt, "CRT  (Cash→Roth→Trad)", "steelblue"),
        (trc, "TRC  (Trad→Roth→Cash)", "darkorange"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(title or "Simulated Sustainable Annual After-Tax Income, by Strategy", fontsize=13)

    for ax, (series, label, color) in zip(axes, panels):
        q_vals = np.percentile(series, pcts)
        med    = float(np.median(series))

        ax.plot(prob, q_vals, color=color, lw=2.5)
        ax.axvline(50, color=color, lw=1.0, linestyle="--", alpha=0.55,
                   label=f"50%: ${med/1_000:.0f}k")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Probability of Sustaining")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x)}%")
        )
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=8)
        _dollar_fmt(ax)

    axes[0].set_ylabel("After-Tax Income")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_marginal_rates(results_df, title=None, save_path=None):
    """
    Plot median ± percentile bands for the top marginal tax rate vs age.
    """
    grp = results_df.groupby("age")["marginal_rate"]
    ages      = np.array(sorted(results_df["age"].unique()))
    med       = grp.median().values
    p10, p25  = grp.quantile(0.10).values, grp.quantile(0.25).values
    p75, p90  = grp.quantile(0.75).values, grp.quantile(0.90).values

    fig, ax = plt.subplots(figsize=(9, 5))
    color = "mediumslateblue"
    ax.fill_between(ages, p10, p90, alpha=0.2, color=color, label="10th–90th pct")
    ax.fill_between(ages, p25, p75, alpha=0.35, color=color, label="25th–75th pct")
    ax.plot(ages, med, color=color, lw=2, label="Median")

    # Bracket reference lines
    brackets = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
    for r in brackets:
        ax.axhline(r, color="gray", lw=0.5, linestyle=":", alpha=0.6)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_yticks(brackets)
    ax.set_title(title or "Top Marginal Tax Rate (Optimal Strategy)")
    ax.set_xlabel("Age")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
