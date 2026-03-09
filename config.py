# config.py — Default parameters and constants

# ── Individual parameters ──────────────────────────────────────────────────
START_AGE = 56
DEATH_AGE = 92
INITIAL_CASH = 500_000        # taxable muni-bond account
INITIAL_RETIREMENT = 2_000_000  # tax-deferred retirement account (IRA/401k)
INITIAL_ROTH = 1_000_000        # Roth IRA (after-tax, tax-free growth and withdrawals)

# Ages
RETIREMENT_AGE = 65           # employment income ends
SS_START_AGE = 70
RMD_START_AGE = 73

# Social Security
SS_ANNUAL = 120_000           # max benefit for couple
SS_TAXABLE_FRAC = 0.85        # fraction of SS subject to ordinary income tax

# Optional pre-retirement employment income (ages 59–retirement_age)
EXTRA_INCOME = 0              # set > 0 to model part-time income

# Optional pension income starting at retirement age (fully taxable)
PENSION_ANNUAL = 0            # set > 0 to model a defined-benefit pension

# Optional IRA contributions during pre-retirement phase (Stage 1 only)
TRAD_IRA_CONTRIB = 0          # annual Traditional IRA contribution
ROTH_IRA_CONTRIB = 0          # annual Roth IRA contribution

# ── Investment parameters ─────────────────────────────────────────────────
MUNI_RATE = 0.03              # fixed annual return on muni bonds
STOCK_MEAN = 0.07             # mean annual stock return
STOCK_STD = 0.15              # std dev of annual stock return
RETIREMENT_STOCK_ALLOC = 0.60  # fraction of traditional IRA invested in stocks
                                # rest is in muni bonds (0.60 stocks / 0.40 bonds)
ROTH_STOCK_ALLOC = 1.0         # Roth: 100% stocks (maximize tax-free compounding)

# ── Simulation ────────────────────────────────────────────────────────────
N_SIMULATIONS = 200
RANDOM_SEED = 42

# ── Objective ─────────────────────────────────────────────────────────────
# "maximize_consumption" : maximize undiscounted lifetime after-tax income (default)
# "minimize_taxes"       : minimize total lifetime taxes paid
OBJECTIVE = "maximize_constant_consumption"

# ── 2026 US tax brackets (Married Filing Jointly) ─────────────────────────
# Format: (lower_bound, upper_bound_or_None, marginal_rate)
# Thresholds are for TAXABLE income (after standard deduction)
BRACKETS_2026_MFJ = [
    (0,        24_800,  0.10),
    (24_800,   100_800, 0.12),
    (100_800,  211_400, 0.22),
    (211_400,  403_550, 0.24),
    (403_550,  512_450, 0.32),
    (512_450,  768_700, 0.35),
    (768_700,  None,    0.37),
]

STANDARD_DEDUCTION_2026_MFJ = 32_200
