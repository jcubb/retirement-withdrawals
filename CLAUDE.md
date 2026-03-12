# retirement-withdrawals — Project Guide for Claude

## Project Overview

A Monte Carlo retirement withdrawal optimizer. Given current age, account balances, income
streams, and market assumptions, the tool finds the maximum constant after-tax annual income
a retiree can sustain until their life expectancy under each simulated stock-return path.

- **Repo:** `jcubb/retirement-withdrawals` on GitHub (branch: `master`)
- **Deployed:** Render (Streamlit web app)
- **Docs:** `paper/` directory — `paper.tex`, `notation.tex`, `paper.pdf`, `notation.pdf`


## Stack

| Library | Role |
|---|---|
| `scipy.optimize.linprog` (HiGHS) | LP solver — default path (~6ms/sim) |
| `cvxpy` + OSQP | QP solver — `minimize_variance_consumption` only (lazy-loaded) |
| `numpy`, `pandas` | numerics and results |
| `matplotlib` | all plots |
| `tqdm` | progress bar in CLI |
| `streamlit` | web UI |


## File Responsibilities

| File | Purpose |
|---|---|
| `config.py` | All default parameters and 2026 MFJ tax brackets |
| `optimizer.py` | LP/QP formulations, benchmark simulations (CRT/TRC) |
| `simulation.py` | Monte Carlo engine; `build_params`, `run_simulations` |
| `analysis.py` | Summary statistics and all matplotlib plots |
| `app.py` | Streamlit UI |
| `tax_utils.py` | `compute_tax_scalar`, `marginal_rate`, `compute_tax_cvxpy` |
| `rmd_utils.py` | IRS Uniform Lifetime Table; `get_distribution_period(age)` |
| `main.py` | CLI entry point |


## 3-Account Model

| Account | Variable | Tax Treatment | Notes |
|---|---|---|---|
| Muni / cash | `w1`, `A1` | Tax-free | Fixed return = `muni_rate`; no RMDs |
| Traditional IRA | `w2`, `A2` | Withdrawals = ordinary income | RMDs from age 73; 10% penalty before 59.5 |
| Roth IRA | `w3`, `A3` | Tax-free | No RMDs; defaults to 100% stocks |

Growth factors per simulation path:
- `H[t]` = `(1 + r_muni)^t` (cash; deterministic)
- `G[t]` = cumulative product of `(1 + r2[t])` (traditional; stochastic blend)
- `G3[t]` = cumulative product of `(1 + r3[t])` (Roth; stochastic blend)


## LP Formulation

**Variables:** `[w1(T), w2(T), w3(T), slack(T × K)]`

Index helpers (inside `optimize_withdrawals`):
- `i1(t) = t`
- `i2(t) = T + t`
- `i3(t) = 2T + t`
- `is_(t, k) = 3T + t*K + k`
- `N = 3T + T*K` (+ 1 extra for `C` in `maximize_constant_consumption`)

**Objectives:**
- `maximize_constant_consumption` (default): adds scalar variable `C`; minimizes `-C`; consumption equality `w1+w2+w3 - Σ dr_k*s_k = C + ss + emp + pension` per year; terminal drain constraints dropped (non-negativity naturally drains accounts)
- `minimize_taxes`: minimize `Σ dr_k * s_k + pen * w2`
- `maximize_consumption`: same cost vector with `w1, w2, w3` negated

**Key constraints:**
- Terminal equality: `Σ w_i(tau) * G[T]/G[tau] = A_i0 * G[T] + contrib_growth[T]`
- Non-negativity: `Σ_{tau<=t} w_i(tau) * G[t]/G[tau] <= A_i0 * G[t] + contrib_growth[t]`
- Tax slack: `w2[t] - s[t,k] <= thresh[k] - ss_tax[t] - emp[t] - pension[t]`
- RMD: `dp * w2[t] + Σ_{tau<t} w2(tau) * G[t]/G[tau] >= A2_0 * G[t] + contrib2_growth[t]`
- Early penalty `pen[t] = 0.10` if `start_age + t < 59.5` else `0.0`

Tax brackets use incremental-rate (delta-rate) formulation; `_DELTA_RATES` and `_THRESHOLDS`
are precomputed at module load from 2026 MFJ brackets in `config.py`.


## IRA Contributions (pre-retirement)

Contributions are modeled as a constant per year during Stage 1 (`t < T_pre`), added to
account balances as constants — they shift constraint RHS values, not decision variables.

**Helper:** `_contrib_accumulated(contrib_annual, T_pre, G, T)` returns array of length `T+1`:
```
result[t] = contrib * sum_{tau=0}^{min(t-1, T_pre-1)} G[t] / G[tau]
```
This modifies: terminal equality RHS, non-negativity RHS, and RMD RHS for A2 and A3.

Config defaults: `TRAD_IRA_CONTRIB = 0`, `ROTH_IRA_CONTRIB = 0`


## Benchmarks

Two perfect-foresight heuristic strategies solved via bisection (`brentq`):

- **CRT** (Cash → Roth → Traditional): draws A1 first, then A3, then A2
- **TRC** (Traditional → Roth → Cash): draws A2 first, then A3, then A1

Each finds the unique constant consumption `C` that depletes all accounts to zero at death.
`_raw_terminal_crt` / `_raw_terminal_trc` track one account uncapped (for sign change in brentq).
Result columns in `results_df`: `consumption_crt`, `consumption_trc`.


## Config Defaults

```python
START_AGE = 56          DEATH_AGE = 92         RETIREMENT_AGE = 65
INITIAL_CASH = 500_000  INITIAL_RETIREMENT = 2_000_000  INITIAL_ROTH = 1_000_000
SS_START_AGE = 70       SS_ANNUAL = 120_000    SS_TAXABLE_FRAC = 0.85
MUNI_RATE = 0.03        STOCK_MEAN = 0.07      STOCK_STD = 0.15
RETIREMENT_STOCK_ALLOC = 0.60    ROTH_STOCK_ALLOC = 1.0
N_SIMULATIONS = 200     RANDOM_SEED = 42
OBJECTIVE = "maximize_constant_consumption"
TRAD_IRA_CONTRIB = 0    ROTH_IRA_CONTRIB = 0
PENSION_ANNUAL = 0      EXTRA_INCOME = 0
```

Tax: 2026 MFJ brackets; `STANDARD_DEDUCTION_2026_MFJ = 32_200`.


## Memory / Performance Notes

- Baseline memory on Render: ~250-350MB (numpy + pandas + scipy + streamlit + matplotlib)
- `cvxpy` (~100-150MB) is **lazy-loaded** in both `optimizer.py` and `tax_utils.py` — only
  imported when `minimize_variance_consumption` is used (not accessible from the UI)
- Render limit: 512MB; the lazy-load fix keeps the app well within bounds
- Speed: ~6ms/sim (LP) → 200 sims in ~1.2s compute time; ~1min wall time on Render (startup + render)


## Known Gotchas

### Streamlit dollar signs in help text
Dollar signs in `help=` strings get rendered as LaTeX math delimiters. Escape them: `\$7,500`.

### LaTeX / paper editing
- **Never** edit `.tex` files via Python string manipulation — `\b`, `\n`, `\r`, `\t` in Python
  string literals silently corrupt `\bar`, `\bigl`, etc. into control characters
- The Edit/Write tools produce `EEXIST mkdir` errors on the `paper/` subdirectory on Windows
- **Best practice:** edit `.tex` files directly in VSCode; compile with `latexmk -pdf file.tex`
  from the `paper/` directory; clean with `latexmk -c`

### Local Streamlit launch (Windows)
The venv path contains spaces, which breaks `run_in_background`. Use:
```
powershell -Command "Start-Process -FilePath 'c:\Users\gcubb\OneDrive\Python\.venv\Scripts\python.exe' -ArgumentList '-m', 'streamlit', 'run', 'app.py', '--server.headless', 'true' -WorkingDirectory 'c:\Users\gcubb\OneDrive\Python\retirement-withdrawals' -WindowStyle Hidden"
```
`--server.headless true` is required (otherwise Streamlit prompts for email and hangs).
Wait ~15 seconds; port defaults to 8501, increments if prior instances still running.
Health check: `netstat -ano | grep 8501` (curl is unreliable in this shell).


## Python Environment

Always use the project venv:
- Run: `C:/Users/gcubb/OneDrive/Python/.venv/Scripts/python.exe <script>`
- Activate (PowerShell): `C:/Users/gcubb/OneDrive/Python/.venv/Scripts/Activate.ps1`
