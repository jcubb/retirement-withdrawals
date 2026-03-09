# simulation.py — Monte Carlo engine

import numpy as np
import pandas as pd
from tqdm import tqdm

from optimizer import optimize_withdrawals, benchmark_consumptions
import config as cfg


def build_params(overrides=None):
    """Build a params dict from config defaults, applying any overrides."""
    params = {
        "start_age":             cfg.START_AGE,
        "death_age":             cfg.DEATH_AGE,
        "initial_cash":          cfg.INITIAL_CASH,
        "initial_retirement":    cfg.INITIAL_RETIREMENT,
        "muni_rate":             cfg.MUNI_RATE,
        "stock_mean":            cfg.STOCK_MEAN,
        "stock_std":             cfg.STOCK_STD,
        "retirement_muni_alloc": 1.0 - cfg.RETIREMENT_STOCK_ALLOC,
        "roth_muni_alloc":       1.0 - cfg.ROTH_STOCK_ALLOC,
        "initial_roth":          cfg.INITIAL_ROTH,
        "ss_annual":             cfg.SS_ANNUAL,
        "ss_start_age":          cfg.SS_START_AGE,
        "ss_taxable_frac":       cfg.SS_TAXABLE_FRAC,
        "extra_income":          cfg.EXTRA_INCOME,
        "pension_annual":        cfg.PENSION_ANNUAL,
        "trad_ira_contrib":      cfg.TRAD_IRA_CONTRIB,
        "roth_ira_contrib":      cfg.ROTH_IRA_CONTRIB,
        "retirement_age":        cfg.RETIREMENT_AGE,
        "n_simulations":         cfg.N_SIMULATIONS,
        "random_seed":           cfg.RANDOM_SEED,
        "objective":             cfg.OBJECTIVE,
    }
    if overrides:
        params.update(overrides)
    return params


def generate_paths(n_sim, T, mean, std, seed=None):
    """
    Generate n_sim independent annual stock-return paths of length T.

    Returns np.ndarray of shape (n_sim, T).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=(n_sim, T))


def run_simulations(params=None, overrides=None, verbose=True, progress_callback=None):
    """
    Run Monte Carlo simulations and return a long-format DataFrame.

    Parameters
    ----------
    params : dict (optional) — if None, built from config + overrides
    overrides : dict (optional) — overrides applied on top of config defaults
    verbose : show tqdm progress bar
    progress_callback : callable(float) or None
        Called after each simulation with the fraction complete (0.0–1.0).
        Use this to drive a Streamlit progress bar.

    Returns
    -------
    results_df : pd.DataFrame with columns
        sim_id, age, w1, w2, w3, A1, A2, A3, income, taxes, consumption, ...
    summary : dict with scalar stats across simulations
    """
    if params is None:
        params = build_params(overrides)

    T = params["death_age"] - params["start_age"]
    paths = generate_paths(
        n_sim=params["n_simulations"],
        T=T,
        mean=params["stock_mean"],
        std=params["stock_std"],
        seed=params["random_seed"],
    )

    records = []
    n_infeasible = 0
    n_total = params["n_simulations"]

    iterator = tqdm(range(n_total), desc="Simulations") if verbose else range(n_total)

    for sim_id in iterator:
        result = optimize_withdrawals(paths[sim_id], params)
        if result is None:
            n_infeasible += 1
            continue

        cons_crt, cons_trc = benchmark_consumptions(paths[sim_id], params)

        if progress_callback is not None:
            progress_callback((sim_id + 1) / n_total)

        ages = np.arange(params["start_age"], params["death_age"])
        for t, age in enumerate(ages):
            records.append({
                "sim_id":          sim_id,
                "age":             age,
                "w1":              result["w1"][t],
                "w2":              result["w2"][t],
                "w3":              result["w3"][t],
                "A1":              result["A1"][t],
                "A2":              result["A2"][t],
                "A3":              result["A3"][t],
                "income":          result["income"][t],
                "taxes":           result["taxes"][t],
                "consumption":     result["consumption"][t],
                "marginal_rate":   result["marginal_rate"][t],
                "consumption_crt": cons_crt[t],
                "consumption_trc": cons_trc[t],
            })

    results_df = pd.DataFrame(records)

    n_solved = params["n_simulations"] - n_infeasible
    summary = {
        "n_simulations": params["n_simulations"],
        "n_solved":      n_solved,
        "n_infeasible":  n_infeasible,
    }
    if n_solved > 0:
        total_taxes = results_df.groupby("sim_id")["taxes"].sum()
        summary["median_total_taxes"]  = total_taxes.median()
        summary["mean_total_taxes"]    = total_taxes.mean()
        summary["p10_total_taxes"]     = total_taxes.quantile(0.10)
        summary["p90_total_taxes"]     = total_taxes.quantile(0.90)

        total_consumption = results_df.groupby("sim_id")["consumption"].sum()
        summary["median_total_consumption"] = total_consumption.median()

        # Median annual consumption (mean within sim, then median across sims)
        annual_consumption = results_df.groupby("sim_id")["consumption"].mean()
        summary["median_annual_consumption"] = annual_consumption.median()

        # Per-sim standard deviation of annual consumption (then median across sims)
        cons_std = results_df.groupby("sim_id")["consumption"].std()
        summary["median_consumption_std"] = cons_std.median()

        # Median total wealth (A1+A2+A3) at retirement age
        ret_age = params["retirement_age"]
        if ret_age > params["start_age"]:
            ret_rows = results_df[results_df["age"] == ret_age].copy()
            ret_rows["total_wealth"] = ret_rows["A1"] + ret_rows["A2"] + ret_rows["A3"]
            summary["median_wealth_at_retirement"] = ret_rows["total_wealth"].median()

    if verbose:
        print(f"\nSolved: {n_solved}/{params['n_simulations']}  |  "
              f"Infeasible: {n_infeasible}")
        if n_solved > 0:
            print(f"Median lifetime taxes:       ${summary['median_total_taxes']:,.0f}")
            print(f"Median total consumption:    ${summary['median_total_consumption']:,.0f}")
            print(f"Median consumption std/year: ${summary['median_consumption_std']:,.0f}")

    return results_df, summary
