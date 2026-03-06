# app.py — Streamlit web app for the retirement withdrawal optimizer

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must precede pyplot imports

import streamlit as st
import matplotlib.pyplot as plt

from simulation import build_params, run_simulations
from analysis import (
    plot_consumption_quantiles,
    plot_account_balances,
    plot_withdrawal_paths,
    plot_marginal_rates,
    plot_tax_distribution,
)
import config as cfg


st.set_page_config(
    page_title="Retirement Planning & Tax Optimizer",
    page_icon="💰",
    layout="wide",
)
st.title("Retirement Planning & Tax Optimizer")

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 10px 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to use this tool", expanded=False):
    st.markdown(
        """
        **Planning horizon and the two stages**

        The tool organizes your financial life into two stages defined by three age inputs:
        your **current age**, your **retirement age**, and your **life expectancy**.

        - **Stage 1 (current age → retirement age):** You are still earning employment income.
          The optimizer can supplement that income with account withdrawals if needed, but the
          primary goal is to let savings continue growing toward retirement.
        - **Stage 2 (retirement age → life expectancy):** Employment income stops. All spending
          must come from the three savings accounts — muni/cash, Traditional IRA, and Roth —
          plus any Pension Income, or Social Security benefits (which begin at the separate SS start age you specify,
          and can fall in either stage).

        The optimizer jointly plans withdrawals across **both** stages simultaneously, so decisions
        made in Stage 1 (such as drawing down the traditional IRA early to reduce future required minimum distribution (RMD)
        exposure) are evaluated against their downstream tax consequences in Stage 2.

        ---

        This tool finds the optimal annual withdrawal strategy from three retirement accounts —
        a **municipal bond (muni/cash)** account, a **Traditional IRA/401(k)**, and a **Roth IRA** —
        across thousands of simulated stock-return scenarios.
        The optimizer chooses how much to draw from each account each year to maximize sustainable
        spending while minimizing lifetime federal income taxes.

        | Source | Tax treatment | Notes |
        |---|---|---|
        | **Muni / Cash** | Federally tax-free | Fixed annual return; no RMDs |
        | **Traditional IRA** | Withdrawals are ordinary income | RMDs start at age 73; 10% penalty before age 59½ |
        | **Roth IRA** | Qualified withdrawals tax-free | No RMDs; no penalty |
        | **Pre-retirement income** | Ordinary income | Earned income before retirement age; phases out at retirement |
        | **Social Security** | 85% included in ordinary income | Begins at SS start age |
        | **Pension** | Ordinary income | Defined-benefit or annuity income; begins at retirement age and continues through the planning horizon |

        **Steps**
        1. Enter your account balances, income details, and return assumptions in the **Setup** sidebar.
        2. Adjust market assumptions and Monte Carlo settings on the **Simulation** tab.
        3. Click **Run Simulation** — results appear on the **Results** tab.

        **Benchmarks:** Results are compared to two simple heuristic strategies —
        **CRT** (Cash → Roth → Traditional) and **TRC** (Traditional → Roth → Cash) —
        which draw from accounts in a fixed priority order and serve as lower bounds on
        what the optimizer achieves.

        **Results**
        
        Some of the key results are contained in the first chart, which shows the probability of being 
        able to sustain different levels of after-tax income when following the optimal withdrawal strategy. 
        The median and 95% worst case values are also shown in the summary table at the top.
        """
    )


# ── Sidebar — setup parameters ────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")

    st.subheader("Ages")
    start_age = st.number_input(
        "Current age", min_value=40, max_value=80, value=cfg.START_AGE,
        help="Your current age. The optimizer plans withdrawals from this age through the life expectancy you set.",
    )
    ret_age = st.number_input(
        "Retirement age", min_value=40, max_value=80, value=cfg.RETIREMENT_AGE,
        help="Age at which earned employment income stops. Pre-retirement income phases out at this age.",
    )
    ss_start = st.number_input(
        "SS start age", min_value=62, max_value=85, value=cfg.SS_START_AGE,
        help="Age at which Social Security payments begin. The model holds your benefit amount fixed at whatever you enter — it does not adjust for claiming age.",
    )
    death_age = st.number_input(
        "Life expectancy", min_value=70, max_value=110, value=cfg.DEATH_AGE,
        help="Planning horizon end. All accounts are drawn to zero by this age. This is a planning assumption, not a prediction.",
    )

    st.subheader("Account Balances")
    initial_cash = st.number_input(
        "Cash / muni-bond ($)",
        min_value=0, max_value=50_000_000, value=cfg.INITIAL_CASH, step=50_000,
        help="Taxable account invested in municipal bonds. Interest and withdrawals are federally tax-free. Earns the fixed muni rate each year.",
    )
    st.caption(f"${initial_cash:,.0f}")
    initial_ret = st.number_input(
        "Traditional IRA / 401k ($)",
        min_value=0, max_value=50_000_000, value=cfg.INITIAL_RETIREMENT, step=50_000,
        help="Traditional IRA or 401(k). All withdrawals are ordinary income and taxed accordingly. Subject to Required Minimum Distributions (RMDs) starting at age 73. Withdrawals before age 59½ carry a 10% early withdrawal penalty.",
    )
    st.caption(f"${initial_ret:,.0f}")
    initial_roth = st.number_input(
        "Roth IRA ($)",
        min_value=0, max_value=50_000_000, value=cfg.INITIAL_ROTH, step=50_000,
        help="Roth IRA. Contributions were made after-tax; all qualified withdrawals (including growth) are tax-free. Not subject to Required Minimum Distributions.",
    )
    st.caption(f"${initial_roth:,.0f}")

    st.subheader("Asset Allocation")
    ret_muni_pct = st.slider(
        "Traditional IRA: muni %", 0, 100, int(cfg.RETIREMENT_MUNI_ALLOC * 100),
        help="Fraction of the Traditional IRA invested in bonds vs. equities. 0% = 100% equities; 40% means a 40/60 bond-equity mix. The blended return is (muni% × muni rate) + (equity% × stock return).",
    )
    roth_muni_pct = st.slider(
        "Roth IRA: muni %", 0, 100, int(cfg.ROTH_MUNI_ALLOC * 100),
        help="Fraction of the Roth IRA invested in bonds vs. equities. Defaults to 0% (all equities) since Roth growth is tax-free and benefits most from higher expected equity returns.",
    )

    st.subheader("Social Security")
    ss_annual = st.number_input(
        "Annual SS benefit ($)",
        min_value=0, max_value=300_000, value=cfg.SS_ANNUAL, step=1_000,
        help="Gross annual Social Security benefit once payments begin. 85% of SS benefits are typically included in taxable income. The maximum combined benefit for a couple who both wait until age 70 is approximately $120,000 as of 2026.",
    )
    st.caption(f"${ss_annual:,.0f}")

    st.subheader("Other Income")
    extra_income = st.number_input(
        "Pre-retirement income ($/yr)",
        min_value=0, max_value=2_000_000, value=cfg.EXTRA_INCOME, step=10_000,
        help="Employment or other earned income received before retirement age. Fully included in taxable ordinary income. Phases out to zero at the retirement age you set.",
    )
    st.caption(f"${extra_income:,.0f}")
    pension_annual = st.number_input(
        "Annual pension income ($/yr)",
        min_value=0, max_value=1_000_000, value=cfg.PENSION_ANNUAL, step=5_000,
        help="Defined-benefit pension or other annuity income that begins at retirement age and continues through the end of the planning horizon. Fully included in taxable ordinary income. Leave at $0 if you have no pension.",
    )
    st.caption(f"${pension_annual:,.0f}")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_sim, tab_results = st.tabs(["▶  Simulation", "📊  Results"])


# ── Simulation tab ────────────────────────────────────────────────────────────
with tab_sim:
    col_mkt, col_mc = st.columns(2)

    with col_mkt:
        st.subheader("Market Assumptions")
        muni_rate_pct = st.number_input(
            "Muni bond rate (%)", min_value=0.0, max_value=20.0,
            value=round(cfg.MUNI_RATE * 100, 1), step=0.1, format="%.1f",
            help="Fixed annual return on municipal bonds. Applied to the muni/cash account and to the bond portion of the IRA accounts.",
        )
        stock_mean_pct = st.number_input(
            "Stock mean return (%)", min_value=-10.0, max_value=30.0,
            value=round(cfg.STOCK_MEAN * 100, 1), step=0.1, format="%.1f",
            help="Expected annual return on the equity portion of the IRA accounts. Stock returns are drawn from a normal distribution with this mean each year.",
        )
        stock_std_pct = st.number_input(
            "Stock volatility (%)", min_value=1.0, max_value=60.0,
            value=round(cfg.STOCK_STD * 100, 1), step=0.5, format="%.1f",
            help="Annual standard deviation of stock returns. A value of 15% is broadly consistent with historical U.S. equity markets. Higher values widen the range of outcomes across simulations.",
        )

    with col_mc:
        st.subheader("Monte Carlo")
        n_sim = st.number_input(
            "Number of simulations", min_value=10, max_value=10_000,
            value=cfg.N_SIMULATIONS, step=100,
            help="Number of independent stock-return paths to simulate. More simulations give more reliable percentile estimates at the cost of longer run time. 1,000 paths typically completes in about 30 seconds.",
        )
        use_seed = st.checkbox("Fix random seed", value=True)
        seed_val = (
            st.number_input("Seed", min_value=0, max_value=99_999, value=cfg.RANDOM_SEED,
                            help="Integer used to initialize the random number generator. Fixing the seed makes results fully reproducible — the same seed always produces the same stock-return paths. Change the seed to see how results vary across different random draws.")
            if use_seed else None
        )

    st.write("")
    run_clicked = st.button("▶  Run Simulation", type="primary")

    if run_clicked:
        params = build_params({
            "start_age":             int(start_age),
            "death_age":             int(death_age),
            "retirement_age":        int(ret_age),
            "ss_start_age":          int(ss_start),
            "initial_cash":          float(initial_cash),
            "initial_retirement":    float(initial_ret),
            "initial_roth":          float(initial_roth),
            "retirement_muni_alloc": ret_muni_pct  / 100.0,
            "roth_muni_alloc":       roth_muni_pct / 100.0,
            "ss_annual":             float(ss_annual),
            "ss_taxable_frac":       cfg.SS_TAXABLE_FRAC,
            "extra_income":          float(extra_income),
            "pension_annual":        float(pension_annual),
            "muni_rate":             muni_rate_pct  / 100.0,
            "stock_mean":            stock_mean_pct / 100.0,
            "stock_std":             stock_std_pct  / 100.0,
            "n_simulations":         int(n_sim),
            "random_seed":           seed_val,
        })

        progress_bar = st.progress(0.0, text="Starting…")
        status_text  = st.empty()

        def _on_progress(frac):
            progress_bar.progress(frac, text=f"Running simulations… {frac * 100:.0f}%")

        results_df, summary = run_simulations(
            params, verbose=False, progress_callback=_on_progress,
        )

        progress_bar.progress(1.0, text="Complete!")
        status_text.success(
            f"Done — {summary['n_solved']:,} / {summary['n_simulations']:,} solved. "
            "Switch to the Results tab to view charts."
        )

        st.session_state["results_df"] = results_df
        st.session_state["summary"]    = summary


# ── Results tab ───────────────────────────────────────────────────────────────
with tab_results:
    if "results_df" not in st.session_state:
        st.info("No results yet — run a simulation on the Simulation tab.")
    else:
        df = st.session_state["results_df"]
        s  = st.session_state["summary"]

        # ── Summary metrics ───────────────────────────────────────────────
        st.subheader("Summary")
        annual_consumption = df.groupby("sim_id")["consumption"].mean()
        p5_annual = annual_consumption.quantile(0.05)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Median wealth at retirement",
                  f"${s.get('median_wealth_at_retirement', 0):,.0f}")
        c2.metric("Median annual after-tax income",
                  f"${s.get('median_annual_consumption', 0):,.0f}")
        c3.metric("95% chance min after-tax income",
                  f"${p5_annual:,.0f}")
        c4.metric("Median lifetime after-tax income",
                  f"${s.get('median_total_consumption', 0):,.0f}")
        c5.metric("Median lifetime taxes",
                  f"${s.get('median_total_taxes', 0):,.0f}")
        c6.metric("Simulations solved",
                  f"{s['n_solved']:,} / {s['n_simulations']:,}")

        st.divider()

        # ── Plots ─────────────────────────────────────────────────────────
        for label, fn in [
            ("After-Tax Income",  plot_consumption_quantiles),
            ("Account Balances",  plot_account_balances),
            ("Withdrawals",               plot_withdrawal_paths),
            ("Marginal Tax Rates",        plot_marginal_rates),
            ("Tax Distribution",          plot_tax_distribution),
        ]:
            st.subheader(label)
            fig = fn(df)
            st.pyplot(fig)
            plt.close(fig)
