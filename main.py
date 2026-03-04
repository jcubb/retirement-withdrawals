# main.py — CLI entry point for retirement withdrawal optimizer

import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the project directory is on the path when running from elsewhere
sys.path.insert(0, os.path.dirname(__file__))

from simulation import run_simulations, build_params
from analysis import (
    summary_stats, lifetime_tax_summary,
    plot_withdrawal_paths, plot_account_balances,
    plot_tax_distribution, plot_consumption, plot_marginal_rates,
)


def parse_args():
    p = argparse.ArgumentParser(description="Retirement withdrawal optimizer")
    p.add_argument("--death-age",         type=int,   default=None)
    p.add_argument("--initial-cash",       type=float, default=None)
    p.add_argument("--initial-retirement", type=float, default=None)
    p.add_argument("--muni-rate",          type=float, default=None)
    p.add_argument("--stock-mean",         type=float, default=None)
    p.add_argument("--stock-std",          type=float, default=None)
    p.add_argument("--muni-alloc",         type=float, default=None,
                   help="Fraction of retirement account in muni bonds (0–1)")
    p.add_argument("--initial-roth",       type=float, default=None)
    p.add_argument("--roth-muni-alloc",    type=float, default=None,
                   help="Fraction of Roth account in muni bonds (0–1)")
    p.add_argument("--extra-income",       type=float, default=None,
                   help="Additional employment income ages 59–65")
    p.add_argument("--n-sim",              type=int,   default=None,
                   help="Number of Monte Carlo simulations")
    p.add_argument("--seed",               type=int,   default=None)
    p.add_argument("--objective",          type=str,   default=None,
                   choices=["maximize_constant_consumption", "maximize_consumption",
                            "minimize_taxes", "minimize_variance_consumption"],
                   help="Optimization objective (default: maximize_constant_consumption)")
    p.add_argument("--out-csv",            type=str,   default="results.csv",
                   help="Output CSV file path")
    p.add_argument("--plots",              action="store_true",
                   help="Show plots after simulation")
    p.add_argument("--save-plots",         type=str,   default=None,
                   help="Directory to save plot images")
    return p.parse_args()


def main():
    args = parse_args()

    # Map CLI args → param overrides (skip None values)
    override_map = {
        "death_age":             args.death_age,
        "initial_cash":          args.initial_cash,
        "initial_retirement":    args.initial_retirement,
        "muni_rate":             args.muni_rate,
        "stock_mean":            args.stock_mean,
        "stock_std":             args.stock_std,
        "retirement_muni_alloc": args.muni_alloc,
        "initial_roth":          args.initial_roth,
        "roth_muni_alloc":       args.roth_muni_alloc,
        "extra_income":          args.extra_income,
        "n_simulations":         args.n_sim,
        "random_seed":           args.seed,
        "objective":             args.objective,
    }
    overrides = {k: v for k, v in override_map.items() if v is not None}
    params = build_params(overrides)

    print("Running simulations with params:")
    for k in ("start_age", "death_age", "initial_cash", "initial_retirement",
              "muni_rate", "stock_mean", "stock_std", "retirement_muni_alloc",
              "extra_income", "n_simulations"):
        print(f"  {k}: {params[k]}")
    print()

    results_df, summary = run_simulations(params=params, verbose=True)

    print(f"\nSummary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: ${v:,.0f}")
        else:
            print(f"  {k}: {v}")

    # Save to CSV
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nResults saved to: {args.out_csv}")

    # Plots
    if args.plots or args.save_plots:
        save_dir = args.save_plots
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        def sp(name):
            return os.path.join(save_dir, name) if save_dir else None

        plot_withdrawal_paths(results_df, save_path=sp("withdrawals.png"))
        plot_account_balances(results_df, save_path=sp("balances.png"))
        plot_tax_distribution(results_df, save_path=sp("tax_dist.png"))
        plot_consumption(results_df, save_path=sp("consumption.png"))
        plot_marginal_rates(results_df, save_path=sp("marginal_rates.png"))

        if args.plots:
            plt.show()


if __name__ == "__main__":
    main()
