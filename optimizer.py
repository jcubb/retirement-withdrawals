# optimizer.py — LP and QP formulations for retirement withdrawal optimization
#
# For LP objectives (minimize_taxes, maximize_consumption):
#   Variables layout: [w1 (T), w2 (T), w3 (T), slack (T × K)]
#     w1 = muni cash (tax-free), w2 = traditional IRA (taxable), w3 = Roth (tax-free)
#   Solver: scipy.optimize.linprog with HiGHS backend (~6ms/sim)
#
# For QP objective (minimize_variance_consumption):
#   Variables: w1 (T), w2 (T), w3 (T), s (T × K) — CVXPY Variables
#   Objective: minimize Σ(c[t] − c̄)²  where c[t] = w1+w2+w3+ss+emp−tax is affine
#   Solver: CVXPY + OSQP (~50–200ms/sim depending on problem difficulty)

import numpy as np
from scipy.optimize import linprog, brentq

from tax_utils import compute_tax_scalar, marginal_rate as _marginal_rate
from rmd_utils import get_distribution_period
from config import (
    BRACKETS_2026_MFJ, STANDARD_DEDUCTION_2026_MFJ, RMD_START_AGE
)


# ── Tax bracket constants (computed once at module load) ──────────────────
_BRACKETS = BRACKETS_2026_MFJ
_STD_DED  = STANDARD_DEDUCTION_2026_MFJ

_K = len(_BRACKETS)
_DELTA_RATES = []
_THRESHOLDS  = []   # lower bound of bracket k plus std deduction
prev_rate = 0.0
for lower, upper, rate in _BRACKETS:
    _DELTA_RATES.append(rate - prev_rate)
    _THRESHOLDS.append(_STD_DED + lower)
    prev_rate = rate
_DELTA_RATES = np.array(_DELTA_RATES)
_THRESHOLDS  = np.array(_THRESHOLDS)


def _contrib_accumulated(contrib_annual, T_pre, G, T):
    """
    Compounded value of a fixed annual contribution added at the start of each
    pre-retirement year (t = 0 .. T_pre-1).

    contrib at year tau grows by G[t]/G[tau] to reach time t, so:
        result[t] = contrib * sum_{tau=0}^{min(t-1, T_pre-1)} G[t] / G[tau]

    Returns array of length T+1 (result[0] = 0 always).
    """
    if contrib_annual == 0.0 or T_pre == 0:
        return np.zeros(T + 1)
    inv_G = 1.0 / G[:T_pre]                     # 1/G[0], 1/G[1], ..., 1/G[T_pre-1]
    inv_cumsum = np.zeros(T_pre + 1)
    inv_cumsum[1:] = np.cumsum(inv_G)            # inv_cumsum[k] = sum_{tau=0}^{k-1} 1/G[tau]
    out = np.zeros(T + 1)
    for t in range(1, T + 1):
        k = min(t, T_pre)
        out[t] = contrib_annual * G[t] * inv_cumsum[k]
    return out


def _solve_variance_qp(ss, emp, ss_tax, pen, pension, G, H, G3, A1_0, A2_0, A3_0,
                       contrib2_growth, contrib3_growth, start_age):
    """
    Solve QP to minimize variance of annual after-tax consumption.

    Consumption in year t:
        c[t] = w1[t] + w2[t] + w3[t] + ss[t] + emp[t]
               - Tax(w2[t] + ss_tax[t] + emp[t])

    w1 = muni cash (tax-free), w2 = traditional IRA (taxable), w3 = Roth (tax-free).

    Objective:  minimize  Σ(c[t] − c̄)²  +  ε·Σs   (ε tiny: pins s to minimum)

    Problem is scaled to units of millions ($M) for OSQP numerical stability.
    Returns (w1_opt, w2_opt, w3_opt) in original dollar units, or None.
    """
    import cvxpy as cp  # lazy import — only loaded when QP objective is used

    T = len(ss)
    K = _K

    sc = 1e6
    ss_s      = ss      / sc
    emp_s     = emp     / sc
    ss_tax_s  = ss_tax  / sc
    pension_s = pension / sc
    A1_s      = A1_0   / sc
    A2_s      = A2_0   / sc
    A3_s      = A3_0   / sc
    thresh_s  = _THRESHOLDS / sc

    w1 = cp.Variable(T, nonneg=True)
    w2 = cp.Variable(T, nonneg=True)
    w3 = cp.Variable(T, nonneg=True)
    s  = cp.Variable((T, K), nonneg=True)

    c_expr = w1 + cp.multiply(1.0 - pen, w2) + w3 + (ss_s + emp_s + pension_s) - (s @ _DELTA_RATES)
    c_mean = cp.sum(c_expr) / T
    obj = cp.Minimize(cp.sum_squares(c_expr - c_mean) + 1e-3 * cp.sum(s))

    h_coeff  = H[T]  / H[:T]
    g_coeff  = G[T]  / G[:T]
    g3_coeff = G3[T] / G3[:T]

    L_H  = np.tril(H[1:T+1][:,  np.newaxis] / H[:T][np.newaxis, :])
    L_G  = np.tril(G[1:T+1][:,  np.newaxis] / G[:T][np.newaxis, :])
    L_G3 = np.tril(G3[1:T+1][:, np.newaxis] / G3[:T][np.newaxis, :])

    c2g_s = contrib2_growth / sc
    c3g_s = contrib3_growth / sc

    constraints = [
        h_coeff  @ w1 == A1_s * H[T],
        g_coeff  @ w2 == A2_s * G[T]   + c2g_s[T],
        g3_coeff @ w3 == A3_s * G3[T]  + c3g_s[T],
        L_H  @ w1 <= A1_s * H[1:T+1],
        L_G  @ w2 <= A2_s * G[1:T+1]  + c2g_s[1:T+1],
        L_G3 @ w3 <= A3_s * G3[1:T+1] + c3g_s[1:T+1],
    ]

    for t in range(T):
        constraints.append(w2[t] - s[t, :] <= thresh_s - ss_tax_s[t] - emp_s[t] - pension_s[t])

    # RMD applies to A2 (traditional) only; Roth has no RMD
    rmd_A, rmd_b = [], []
    for t in range(T):
        dp = get_distribution_period(start_age + t)
        if dp is None:
            continue
        row = np.zeros(T)
        row[t] = dp
        for tau in range(t):
            row[tau] = G[t] / G[tau]
        rmd_A.append(row)
        rmd_b.append(A2_s * G[t] + c2g_s[t])
    if rmd_A:
        constraints.append(np.array(rmd_A) @ w2 >= np.array(rmd_b))

    prob = cp.Problem(obj, constraints)
    prob.solve(
        solver=cp.OSQP,
        warm_start=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        verbose=False,
        max_iter=50_000,
    )

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT):
        return None
    if w1.value is None or w2.value is None or w3.value is None:
        return None

    return (
        np.maximum(0.0, w1.value) * sc,
        np.maximum(0.0, w2.value) * sc,
        np.maximum(0.0, w3.value) * sc,
    )


def optimize_withdrawals(r_stock_path, params):
    """
    Find the optimal withdrawal schedule for one simulated stock-return path.

    Parameters
    ----------
    r_stock_path : np.ndarray, shape (T,)
        Realized annual stock returns for years 0..T-1.
    params : dict  (see simulation.build_params for keys)

    Returns
    -------
    dict with arrays (length T and T+1) and scalars, or None if infeasible.
    """
    start_age = params["start_age"]
    death_age = params["death_age"]
    A1_0      = params["initial_cash"]
    A2_0      = params["initial_retirement"]
    r_muni    = params["muni_rate"]
    alpha     = params["retirement_muni_alloc"]
    ss_annual = params["ss_annual"]
    ss_start  = params["ss_start_age"]
    ss_frac   = params["ss_taxable_frac"]
    extra_inc = params["extra_income"]
    ret_age   = params["retirement_age"]

    objective = params.get("objective", "maximize_consumption")

    T = death_age - start_age
    K = _K

    pension_annual  = params.get("pension_annual",   0.0)
    trad_contrib    = params.get("trad_ira_contrib",  0.0)
    roth_contrib    = params.get("roth_ira_contrib",  0.0)
    T_pre           = max(0, min(ret_age - start_age, T))

    # ── Exogenous income arrays ───────────────────────────────────────────
    ss      = np.array([ss_annual     if (start_age + t) >= ss_start else 0.0 for t in range(T)])
    emp     = np.array([extra_inc     if (start_age + t) < ret_age   else 0.0 for t in range(T)])
    pension = np.array([pension_annual if (start_age + t) >= ret_age  else 0.0 for t in range(T)])
    ss_tax  = ss_frac * ss

    # Early withdrawal penalty: 10% on traditional IRA withdrawals before age 59.5
    pen = np.array([0.10 if (start_age + t) < 59.5 else 0.0 for t in range(T)])

    # ── Retirement account cumulative growth factors ──────────────────────
    r2 = alpha * r_muni + (1.0 - alpha) * r_stock_path   # shape (T,)
    G  = np.ones(T + 1)
    for t in range(T):
        G[t + 1] = G[t] * (1.0 + r2[t])

    # Cash account growth factor (deterministic): H[t] = (1 + r_muni)^t
    H = (1.0 + r_muni) ** np.arange(T + 1)

    # ── Roth account growth factors ────────────────────────────────────────
    alpha3 = params.get("roth_muni_alloc", 0.0)
    A3_0   = params.get("initial_roth", 0.0)
    r3     = alpha3 * r_muni + (1.0 - alpha3) * r_stock_path
    G3     = np.ones(T + 1)
    for t in range(T):
        G3[t + 1] = G3[t] * (1.0 + r3[t])

    # ── IRA contribution growth arrays ─────────────────────────────────────
    contrib2_growth = _contrib_accumulated(trad_contrib, T_pre, G,  T)
    contrib3_growth = _contrib_accumulated(roth_contrib,  T_pre, G3, T)
    # Per-year contribution arrays for result extraction
    c2 = np.array([trad_contrib if t < T_pre else 0.0 for t in range(T)])
    c3 = np.array([roth_contrib  if t < T_pre else 0.0 for t in range(T)])

    # ── QP branch: minimize variance of annual consumption ────────────────
    if objective == "minimize_variance_consumption":
        qp_out = _solve_variance_qp(ss, emp, ss_tax, pen, pension, G, H, G3, A1_0, A2_0, A3_0,
                                    contrib2_growth, contrib3_growth, start_age)
        if qp_out is None:
            return None
        w1_opt, w2_opt, w3_opt = qp_out

    # ── LP branch ─────────────────────────────────────────────────────────
    else:
        def i1(t):     return t
        def i2(t):     return T + t
        def i3(t):     return 2 * T + t
        def is_(t, k): return 3 * T + t * K + k
        N = 3 * T + T * K

        # ── Constraints common to all LP objectives ───────────────────────

        # Terminal drain: A1[T] = 0, A2[T] = 0, A3[T] = 0
        # Used as hard equality for minimize_taxes / maximize_consumption.
        # For maximize_constant_consumption the equality is dropped (see below):
        # the balance non-negativity constraints already enforce A[T] >= 0,
        # and the maximize-C objective naturally drains accounts; on extreme
        # paths any residual at death is a bequest rather than infeasibility.
        A_eq = np.zeros((3, N))
        b_eq = np.zeros(3)
        for t in range(T):
            A_eq[0, i1(t)] = H[T] / H[t]
        b_eq[0] = A1_0 * H[T]
        for t in range(T):
            A_eq[1, i2(t)] = G[T] / G[t]
        b_eq[1] = A2_0 * G[T] + contrib2_growth[T]
        for t in range(T):
            A_eq[2, i3(t)] = G3[T] / G3[t]
        b_eq[2] = A3_0 * G3[T] + contrib3_growth[T]

        ub_rows, ub_rhs = [], []

        # Tax slack: w2[t] - s[t,k] <= thresh[k] - ss_tax[t] - emp[t]
        for t in range(T):
            for k in range(K):
                row = np.zeros(N)
                row[i2(t)]     =  1.0
                row[is_(t, k)] = -1.0
                ub_rows.append(row)
                ub_rhs.append(_THRESHOLDS[k] - ss_tax[t] - emp[t] - pension[t])

        # A1[t] >= 0
        for t in range(1, T + 1):
            row = np.zeros(N)
            for tau in range(t):
                row[i1(tau)] = H[t] / H[tau]
            ub_rows.append(row)
            ub_rhs.append(A1_0 * H[t])

        # A2[t] >= 0
        for t in range(1, T + 1):
            row = np.zeros(N)
            for tau in range(t):
                row[i2(tau)] = G[t] / G[tau]
            ub_rows.append(row)
            ub_rhs.append(A2_0 * G[t] + contrib2_growth[t])

        # A3[t] >= 0
        for t in range(1, T + 1):
            row = np.zeros(N)
            for tau in range(t):
                row[i3(tau)] = G3[t] / G3[tau]
            ub_rows.append(row)
            ub_rhs.append(A3_0 * G3[t] + contrib3_growth[t])

        # RMD (traditional account only; Roth has no RMD)
        for t in range(T):
            age = start_age + t
            dp  = get_distribution_period(age)
            if dp is None:
                continue
            row = np.zeros(N)
            row[i2(t)] = -dp
            for tau in range(t):
                row[i2(tau)] = -G[t] / G[tau]
            ub_rows.append(row)
            ub_rhs.append(-A2_0 * G[t] - contrib2_growth[t])

        A_ub = np.array(ub_rows) if ub_rows else np.empty((0, N))
        b_ub = np.array(ub_rhs)  if ub_rhs  else np.empty(0)

        # ── maximize_constant_consumption: add scalar C, T equality rows ──
        if objective == "maximize_constant_consumption":
            # One extra variable C at position N; objective: minimize -C
            i_C = N
            c_obj = np.zeros(N + 1)
            c_obj[i_C] = -1.0

            A_ub_e = np.hstack([A_ub, np.zeros((len(A_ub), 1))])

            # T rows: w1[t]+w2[t]+w3[t] - Σ_k dr[k]*s[t,k] - C = -(ss[t]+emp[t])
            cons_A = np.zeros((T, N + 1))
            for t in range(T):
                cons_A[t, i1(t)]  =  1.0
                cons_A[t, i2(t)]  =  1.0 - pen[t]   # net after early withdrawal penalty
                cons_A[t, i3(t)]  =  1.0
                for k in range(K):
                    cons_A[t, is_(t, k)] = -_DELTA_RATES[k]
                cons_A[t, i_C] = -1.0
            cons_b = -(ss + emp + pension)   # shape (T,)

            # Terminal equality dropped: A[T] >= 0 already enforced by balance
            # constraints; maximize-C naturally drains accounts.
            result = linprog(
                c=c_obj,
                A_ub=A_ub_e, b_ub=b_ub,
                A_eq=cons_A,
                b_eq=cons_b,
                bounds=[(0.0, None)] * N + [(None, None)],
                method="highs",
            )

        # ── minimize_taxes / maximize_consumption ─────────────────────────
        else:
            c_obj = np.zeros(N)
            for t in range(T):
                for k in range(K):
                    c_obj[is_(t, k)] = _DELTA_RATES[k]
                c_obj[i2(t)] = pen[t]   # early withdrawal penalty on traditional IRA
            if objective == "maximize_consumption":
                for t in range(T):
                    c_obj[i1(t)] = -1.0
                    c_obj[i2(t)] -= 1.0   # net: -(1 - pen[t])
                    c_obj[i3(t)] = -1.0

            result = linprog(
                c=c_obj,
                A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq,
                bounds=[(0.0, None)] * N,
                method="highs",
            )

        if result.status not in (0, 1):   # 0=optimal, 1=optimal (iteration limit)
            return None

        x      = result.x
        w1_opt = x[:T]
        w2_opt = x[T:2 * T]
        w3_opt = x[2 * T:3 * T]

    # ── Extract results (common to both branches) ─────────────────────────
    A1_path = np.zeros(T + 1)
    A2_path = np.zeros(T + 1)
    A3_path = np.zeros(T + 1)
    A1_path[0] = A1_0
    A2_path[0] = A2_0
    A3_path[0] = A3_0
    for t in range(T):
        A1_path[t + 1] = (A1_path[t] - w1_opt[t])           * (1.0 + r_muni)
        A2_path[t + 1] = (A2_path[t] - w2_opt[t] + c2[t])   * (1.0 + r2[t])
        A3_path[t + 1] = (A3_path[t] - w3_opt[t] + c3[t])   * (1.0 + r3[t])

    income_tax = np.array([
        compute_tax_scalar(w2_opt[t] + ss_tax[t] + emp[t] + pension[t])
        for t in range(T)
    ])
    taxes       = income_tax + pen * w2_opt   # income tax + early withdrawal penalty
    income      = w2_opt + ss_tax + emp + pension
    consumption = w1_opt + w2_opt + w3_opt + ss + emp + pension - taxes
    marg_rates  = np.array([
        _marginal_rate(w2_opt[t] + ss_tax[t] + emp[t] + pension[t])
        for t in range(T)
    ])

    return {
        "w1":            w1_opt,
        "w2":            w2_opt,
        "w3":            w3_opt,
        "A1":            A1_path,
        "A2":            A2_path,
        "A3":            A3_path,
        "income":        income,
        "taxes":         taxes,
        "consumption":   consumption,
        "marginal_rate": marg_rates,
        "total_taxes":   taxes.sum(),
        "status":        "optimal",
    }


def _sim_crt(C, ss, emp, ss_tax, pen, pension, r_muni, r2, r3, A1_0, A2_0, A3_0,
             contrib2, contrib3):
    """
    Simulate Cash → Roth → Traditional ordering with constant consumption target C.

    Draw from A1 (muni, tax-free) first, then A3 (Roth, tax-free), then A2
    (traditional, taxable).  All account balances are floored at zero.
    Returns (consumption_array, A1_terminal, A2_terminal, A3_terminal).
    """
    T = len(ss)
    A1, A2, A3 = float(A1_0), float(A2_0), float(A3_0)
    cons = np.zeros(T)

    for t in range(T):
        st, et, sst, pt = ss[t], emp[t], ss_tax[t], pension[t]
        pen_t = pen[t]

        # Amount needed from tax-free sources when w2=0
        tax0 = compute_tax_scalar(sst + et + pt)
        needed_tf = C - st - et - pt + tax0   # w1 + w3 required if w2 = 0

        if needed_tf <= A1:
            w1 = max(0.0, needed_tf)
            w3 = 0.0
            w2 = 0.0
        elif needed_tf <= A1 + A3:
            w1 = A1
            w3 = needed_tf - A1
            w2 = 0.0
        else:
            w1 = A1
            w3 = A3
            target = C - w1 - w3 - st - et - pt
            def f(x, pen_t=pen_t, pt=pt):
                return x * (1 - pen_t) - compute_tax_scalar(x + sst + et + pt) - target
            w2 = (0.0 if f(0.0) >= 0.0 else
                  A2  if f(A2)  <= 0.0 else
                  brentq(f, 0.0, A2, xtol=1.0))

        income_tax = compute_tax_scalar(w2 + sst + et + pt)
        taxes   = income_tax + pen_t * w2
        cons[t] = w1 + w3 + w2 + st + et + pt - taxes
        A1 = max(0.0, (A1 - w1)             * (1.0 + r_muni))
        A3 = max(0.0, (A3 - w3 + contrib3[t]) * (1.0 + r3[t]))
        A2 = max(0.0, (A2 - w2 + contrib2[t]) * (1.0 + r2[t]))

    return cons, A1, A2, A3


def _sim_trc(C, ss, emp, ss_tax, pen, pension, r_muni, r2, r3, A1_0, A2_0, A3_0,
             contrib2, contrib3):
    """
    Simulate Traditional → Roth → Cash ordering with constant consumption target C.

    Draw from A2 (traditional, taxable) first, then A3 (Roth, tax-free), then
    A1 (muni, tax-free).  All account balances are floored at zero.
    Returns (consumption_array, A1_terminal, A2_terminal, A3_terminal).
    """
    T = len(ss)
    A1, A2, A3 = float(A1_0), float(A2_0), float(A3_0)
    cons = np.zeros(T)

    for t in range(T):
        st, et, sst, pt = ss[t], emp[t], ss_tax[t], pension[t]
        pen_t = pen[t]

        def f_w2(x, pen_t=pen_t, pt=pt):
            return x * (1 - pen_t) + st + et + pt - compute_tax_scalar(x + sst + et + pt) - C

        if f_w2(0.0) >= 0.0:
            # SS + emp + pension alone covers C; no withdrawals needed
            w2 = 0.0
            w3 = 0.0
            w1 = 0.0
        elif f_w2(A2) <= 0.0:
            # A2 exhausted; supplement from Roth then cash
            w2 = A2
            cons_trad = A2 * (1 - pen_t) + st + et + pt - compute_tax_scalar(A2 + sst + et + pt)
            remaining = C - cons_trad
            if remaining <= 0.0:
                w3 = 0.0
                w1 = 0.0
            elif remaining <= A3:
                w3 = remaining
                w1 = 0.0
            else:
                w3 = A3
                w1 = min(A1, max(0.0, remaining - A3))
        else:
            w2 = brentq(f_w2, 0.0, A2, xtol=1.0)
            w3 = 0.0
            w1 = 0.0

        income_tax = compute_tax_scalar(w2 + sst + et + pt)
        taxes   = income_tax + pen_t * w2
        cons[t] = w1 + w3 + w2 + st + et + pt - taxes
        A2 = max(0.0, (A2 - w2 + contrib2[t]) * (1.0 + r2[t]))
        A3 = max(0.0, (A3 - w3 + contrib3[t]) * (1.0 + r3[t]))
        A1 = max(0.0, (A1 - w1)               * (1.0 + r_muni))

    return cons, A1, A2, A3


def _raw_terminal_crt(C, ss, emp, ss_tax, pen, pension, r_muni, r2, r3, A1_0, A2_0, A3_0,
                      contrib2, contrib3):
    """
    Uncapped terminal A2 balance for the CRT strategy (for bisection).

    A1 and A3 are floored (determine phase switches); A2 is tracked without a
    floor and goes negative when C is too large, providing the sign change
    needed by brentq.  In phase 3, w2 is solved without an upper bound.
    """
    T = len(ss)
    A1     = float(A1_0)
    A3     = float(A3_0)
    A2_raw = float(A2_0)   # uncapped

    for t in range(T):
        st, et, sst, pt = ss[t], emp[t], ss_tax[t], pension[t]
        A1_avail = max(0.0, A1)
        A3_avail = max(0.0, A3)
        pen_t = pen[t]

        tax0      = compute_tax_scalar(sst + et + pt)
        needed_tf = C - st - et - pt + tax0

        if needed_tf <= A1_avail:
            w1 = max(0.0, needed_tf)
            w3 = 0.0
            w2 = 0.0
        elif needed_tf <= A1_avail + A3_avail:
            w1 = A1_avail
            w3 = needed_tf - A1_avail
            w2 = 0.0
        else:
            w1     = A1_avail
            w3     = A3_avail
            target = C - w1 - w3 - st - et - pt
            def f(x, pen_t=pen_t, pt=pt):
                return x * (1 - pen_t) - compute_tax_scalar(x + sst + et + pt) - target
            if f(0.0) >= 0.0:
                w2 = 0.0
            else:
                w2_hi = max(10.0 * C, 1e7)
                w2 = brentq(f, 0.0, w2_hi, xtol=1.0)

        A1     = max(0.0, (A1     - w1)               * (1.0 + r_muni))
        A3     = max(0.0, (A3     - w3 + contrib3[t])  * (1.0 + r3[t]))
        A2_raw =          (A2_raw - w2 + contrib2[t])  * (1.0 + r2[t])   # no floor

    return A2_raw   # > 0 if surplus, < 0 if overdraft


def _raw_terminal_trc(C, ss, emp, ss_tax, pen, pension, r_muni, r2, r3, A1_0, A2_0, A3_0,
                      contrib2, contrib3):
    """
    Uncapped terminal A1 balance for the TRC strategy (for bisection).

    A2 and A3 are floored (determine phase switches); A1 is tracked without a
    floor and goes negative when C is too large.
    """
    T = len(ss)
    A2     = float(A2_0)
    A3     = float(A3_0)
    A1_raw = float(A1_0)   # uncapped

    for t in range(T):
        st, et, sst, pt = ss[t], emp[t], ss_tax[t], pension[t]
        A2_avail = max(0.0, A2)
        A3_avail = max(0.0, A3)
        pen_t = pen[t]

        def f_w2(x, pen_t=pen_t, pt=pt):
            return x * (1 - pen_t) + st + et + pt - compute_tax_scalar(x + sst + et + pt) - C

        if f_w2(0.0) >= 0.0:
            w2 = 0.0
            w3 = 0.0
            w1 = 0.0
        elif f_w2(A2_avail) <= 0.0:
            w2 = A2_avail
            cons_trad = A2_avail * (1 - pen_t) + st + et + pt - compute_tax_scalar(A2_avail + sst + et + pt)
            remaining = C - cons_trad
            if remaining <= 0.0:
                w3 = 0.0
                w1 = 0.0
            elif remaining <= A3_avail:
                w3 = remaining
                w1 = 0.0
            else:
                w3 = A3_avail
                w1 = max(0.0, remaining - A3_avail)   # unconstrained (A1_raw can go negative)
        else:
            w2 = brentq(f_w2, 0.0, A2_avail, xtol=1.0)
            w3 = 0.0
            w1 = 0.0

        A2     = max(0.0, (A2     - w2 + contrib2[t]) * (1.0 + r2[t]))
        A3     = max(0.0, (A3     - w3 + contrib3[t]) * (1.0 + r3[t]))
        A1_raw =          (A1_raw - w1)                * (1.0 + r_muni)   # no floor

    return A1_raw   # > 0 if surplus, < 0 if overdraft


def benchmark_consumptions(r_stock_path, params):
    """
    Compute constant-consumption benchmark strategies with perfect foresight.

    For each 3-account ordering strategy, finds the unique constant annual
    consumption C that depletes all three accounts to zero at the death age.

    CRT (Cash → Roth → Traditional):
        Draw from muni (A1) first, then Roth (A3), then traditional (A2).
        Find C such that the uncapped A2 terminal balance = 0.

    TRC (Traditional → Roth → Cash):
        Draw from traditional (A2) first, then Roth (A3), then muni (A1).
        Find C such that the uncapped A1 terminal balance = 0.

    Parameters
    ----------
    r_stock_path : np.ndarray, shape (T,)
    params : dict

    Returns
    -------
    (cons_crt, cons_trc) : two np.ndarrays of shape (T,)
    """
    start_age = params["start_age"]
    death_age = params["death_age"]
    A1_0      = params["initial_cash"]
    A2_0      = params["initial_retirement"]
    A3_0      = params.get("initial_roth", 0.0)
    r_muni    = params["muni_rate"]
    alpha     = params["retirement_muni_alloc"]
    alpha3    = params.get("roth_muni_alloc", 0.0)
    ss_annual = params["ss_annual"]
    ss_start  = params["ss_start_age"]
    ss_frac   = params["ss_taxable_frac"]
    extra_inc = params["extra_income"]
    ret_age   = params["retirement_age"]

    pension_annual = params.get("pension_annual", 0.0)
    trad_contrib   = params.get("trad_ira_contrib", 0.0)
    roth_contrib   = params.get("roth_ira_contrib",  0.0)

    T = death_age - start_age
    T_pre   = max(0, min(ret_age - start_age, T))
    ss      = np.array([ss_annual     if (start_age + t) >= ss_start else 0.0 for t in range(T)])
    emp     = np.array([extra_inc     if (start_age + t) < ret_age   else 0.0 for t in range(T)])
    pension = np.array([pension_annual if (start_age + t) >= ret_age  else 0.0 for t in range(T)])
    ss_tax  = ss_frac * ss
    pen     = np.array([0.10 if (start_age + t) < 59.5 else 0.0 for t in range(T)])
    r2      = alpha  * r_muni + (1.0 - alpha)  * r_stock_path
    r3      = alpha3 * r_muni + (1.0 - alpha3) * r_stock_path

    # Cumulative growth factors needed for _contrib_accumulated
    G2 = np.ones(T + 1)
    G3v = np.ones(T + 1)
    for t in range(T):
        G2[t + 1]  = G2[t]  * (1.0 + r2[t])
        G3v[t + 1] = G3v[t] * (1.0 + r3[t])

    contrib2 = np.array([trad_contrib if t < T_pre else 0.0 for t in range(T)])
    contrib3 = np.array([roth_contrib  if t < T_pre else 0.0 for t in range(T)])

    kw = dict(ss=ss, emp=emp, ss_tax=ss_tax, pen=pen, pension=pension,
              r_muni=r_muni, r2=r2, r3=r3, A1_0=A1_0, A2_0=A2_0, A3_0=A3_0,
              contrib2=contrib2, contrib3=contrib3)

    def find_C(terminal_fn):
        """Find C where terminal_fn crosses zero (monotone decreasing in C)."""
        C_hi = (A1_0 + A2_0 + A3_0) / T
        for _ in range(50):
            if terminal_fn(C_hi) < 0.0:
                break
            C_hi *= 2.0
        return brentq(terminal_fn, 0.0, C_hi, xtol=1.0)

    C_crt = find_C(lambda C: _raw_terminal_crt(C, **kw))
    cons_crt, _, _, _ = _sim_crt(C_crt, **kw)

    C_trc = find_C(lambda C: _raw_terminal_trc(C, **kw))
    cons_trc, _, _, _ = _sim_trc(C_trc, **kw)

    return cons_crt, cons_trc
