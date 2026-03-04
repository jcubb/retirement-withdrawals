# rmd_utils.py — IRS Uniform Lifetime Table and Required Minimum Distribution helpers

# Distribution periods from the IRS Uniform Lifetime Table
# (source: retirement-prompt.txt, ages 60–120+)
# RMDs are required starting at age 73.
_DIST_PERIOD = {
    60: 27.1, 61: 26.2, 62: 25.4, 63: 24.5, 64: 23.7,
    65: 22.9, 66: 22.0, 67: 21.2, 68: 20.4, 69: 19.6,
    70: 18.8, 71: 18.0, 72: 17.2, 73: 16.4, 74: 15.6,
    75: 14.8, 76: 14.1, 77: 13.3, 78: 12.6, 79: 11.9,
    80: 11.2, 81: 10.5, 82:  9.9, 83:  9.3, 84:  8.7,
    85:  8.1, 86:  7.6, 87:  7.1, 88:  6.6, 89:  6.1,
    90:  5.7, 91:  5.3, 92:  4.9, 93:  4.6, 94:  4.3,
    95:  4.0, 96:  3.7, 97:  3.4, 98:  3.2, 99:  3.0,
   100:  2.8,101:  2.6,102:  2.5,103:  2.3,104:  2.2,
   105:  2.1,106:  2.1,107:  2.1,108:  2.0,109:  2.0,
   110:  2.0,111:  2.0,112:  2.0,113:  1.9,114:  1.9,
   115:  1.8,116:  1.8,117:  1.6,118:  1.4,119:  1.1,
}
_DIST_PERIOD_120_PLUS = 1.0

RMD_START_AGE = 73


def get_distribution_period(age):
    """
    Return the IRS distribution period for a given age.
    Returns None if age < RMD_START_AGE (no RMD required).
    """
    if age < RMD_START_AGE:
        return None
    if age >= 120:
        return _DIST_PERIOD_120_PLUS
    return _DIST_PERIOD.get(age)


def compute_rmd(balance, age):
    """
    Compute the Required Minimum Distribution for a given year-start balance and age.
    Returns 0.0 if no RMD is required (age < 73) or balance <= 0.
    """
    dp = get_distribution_period(age)
    if dp is None or balance <= 0:
        return 0.0
    return balance / dp
