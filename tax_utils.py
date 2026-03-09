# tax_utils.py — Federal income tax calculation (scalar and CVXPY)

from config import BRACKETS_2026_MFJ, STANDARD_DEDUCTION_2026_MFJ


def compute_tax_scalar(gross_income, brackets=None, std_ded=None):
    """
    Compute federal income tax for a given gross income (scalar).

    gross_income : total ordinary income before standard deduction
    Returns total tax owed (float).
    """
    if brackets is None:
        brackets = BRACKETS_2026_MFJ
    if std_ded is None:
        std_ded = STANDARD_DEDUCTION_2026_MFJ

    taxable = max(0.0, gross_income - std_ded)
    tax = 0.0
    for lower, upper, rate in brackets:
        if taxable <= lower:
            break
        income_in_bracket = taxable - lower if upper is None else min(taxable, upper) - lower
        tax += rate * income_in_bracket
    return tax


def marginal_rate(gross_income, brackets=None, std_ded=None):
    """
    Return the top marginal tax rate for the given gross income.

    Returns the rate of the highest bracket that taxable income reaches,
    or 0.0 if income is below the standard deduction.
    """
    if brackets is None:
        brackets = BRACKETS_2026_MFJ
    if std_ded is None:
        std_ded = STANDARD_DEDUCTION_2026_MFJ

    taxable = max(0.0, gross_income - std_ded)
    if taxable == 0.0:
        return 0.0
    rate = brackets[0][2]
    for lower, _upper, r in brackets:
        if taxable > lower:
            rate = r
        else:
            break
    return rate


def effective_tax_rate(gross_income, **kwargs):
    """Return effective (average) tax rate for a given gross income."""
    if gross_income <= 0:
        return 0.0
    return compute_tax_scalar(gross_income, **kwargs) / gross_income


def compute_tax_cvxpy(income_exprs, brackets=None, std_ded=None):
    """
    Build a CVXPY expression for TOTAL taxes across all years.

    Uses the incremental-rate method:
        T(I) = Σ_k  Δrate_k * cp.pos(I - (std_ded + L_k))
    where Δrate_k = rate_k - rate_{k-1} and L_k = lower bound of bracket k.
    Each cp.pos() term is convex, so the sum is convex — valid for cp.Minimize.

    income_exprs : list of T CVXPY expressions (one per year), each representing
                   the year's total ordinary income BEFORE standard deduction.
    Returns a scalar CVXPY expression for the sum of taxes over all years.
    """
    if brackets is None:
        brackets = BRACKETS_2026_MFJ
    if std_ded is None:
        std_ded = STANDARD_DEDUCTION_2026_MFJ

    # Build incremental (marginal rate delta) list
    increments = []
    prev_rate = 0.0
    for lower, _upper, rate in brackets:
        increments.append((lower, rate - prev_rate))
        prev_rate = rate

    import cvxpy as cp  # lazy import — only loaded when QP objective is used

    total_tax = 0.0
    for income in income_exprs:
        for lower, delta_rate in increments:
            threshold = std_ded + lower
            total_tax = total_tax + delta_rate * cp.pos(income - threshold)

    return total_tax
