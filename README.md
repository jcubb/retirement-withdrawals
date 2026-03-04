# Retirement Withdrawal Optimizer

Monte Carlo simulation + LP optimization engine that finds tax-minimizing withdrawal
strategies from two retirement accounts. Both accounts must reach exactly $0 at the
death age. The problem is solved as a Linear Program for each simulated stock-return path.

---

## Quick Start

```bash
# Run with defaults from config.py, save plots to current folder
C:/Users/gcubb/OneDrive/Python/.venv/Scripts/python.exe main.py --save-plots .

# Override any parameter from the command line
python main.py --death-age 95 --initial-cash 3000000 --n-sim 5000 --save-plots .
```

---

## Accounts

| Account | Tax treatment | Returns |
|---|---|---|
| Cash (muni bond) | Withdrawals and interest are federally tax-free | Fixed `muni_rate` |
| Retirement (IRA/401k) | All withdrawals are ordinary income | `alpha * muni_rate + (1-alpha) * r_stock` |

Stocks draw from `Normal(stock_mean, stock_std)` each year.

---

## Mathematical Model

**Decision variables** (per simulation path, T = death_age − start_age years):
- `w1[t]` — withdrawal from cash account in year t
- `w2[t]` — withdrawal from retirement account in year t

**Account dynamics:**
```
A1[t+1] = (A1[t] - w1[t]) * (1 + r_muni)
A2[t+1] = (A2[t] - w2[t]) * (1 + r2[t])
```

**Taxable ordinary income:**
```
income[t] = w2[t] + 0.85 * ss[t] + emp[t]
```

**Tax function** — incremental-rate LP formulation:
```
Tax(income) = Σ_k  Δrate_k * max(0, income − (std_deduction + bracket_lower_k))
```

**Constraints:**
1. `A1[T] = 0`, `A2[T] = 0` — both accounts exhaust at death
2. `A1[t] ≥ 0`, `A2[t] ≥ 0` — no overdraft
3. `w1[t] ≥ 0`, `w2[t] ≥ 0` — withdrawals only (no deposits)
4. `w2[t] ≥ A2[t] / df[age]` for age ≥ 73 — Required Minimum Distributions

**Objective:** Minimize `Σ_t Tax(income[t])`

This is a pure **Linear Program** (≈280 variables, ≈400 constraints). Solved via
`scipy.optimize.linprog` with the HiGHS backend at ~6–10 ms per simulation.

---

## 2026 MFJ Tax Brackets (in `config.py`)

| Taxable Income | Rate |
|---|---|
| $0 – $24,800 | 10% |
| $24,801 – $100,800 | 12% |
| $100,801 – $211,400 | 22% |
| $211,401 – $403,550 | 24% |
| $403,551 – $512,450 | 32% |
| $512,451 – $768,700 | 35% |
| Over $768,700 | 37% |

Standard deduction (MFJ 2026): $32,200

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All parameters and constants (edit here to change assumptions) |
| `tax_utils.py` | 2026 MFJ tax calculator (scalar + CVXPY expressions) |
| `rmd_utils.py` | IRS Uniform Lifetime Table (ages 60–120+), RMD computation |
| `optimizer.py` | Direct LP formulation via scipy/HiGHS for one simulation path |
| `simulation.py` | Monte Carlo engine — generates stock paths, runs optimizer for each |
| `analysis.py` | Summary stats and four plot types |
| `main.py` | CLI entry point with argparse |

---

## Key Design Notes

- **Consumption = withdrawals + SS + employment income − taxes.** No fixed spending
  target; the optimizer chooses the timing of withdrawals to minimize taxes while
  exhausting both accounts at death.
- **Multiple optima for w1**: cash account withdrawals are tax-free regardless of
  timing, so the LP bunches them at the last year of life. The `w2` path (retirement
  account) is unique and economically meaningful.
- **RMD strategy**: the optimizer typically drains the retirement account before age 73
  by filling up the 10–12% brackets in pre-SS years, avoiding a forced high-bracket
  RMD later. This is the "RMD tax tsunami" avoidance strategy.

---

## Phase 2 (planned)

- GitHub repo (`jcubb` account)
- Streamlit app with interactive parameter UI and result charts
- Deploy to Render (free tier Python web service)
