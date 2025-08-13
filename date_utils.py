# date_utils.py
"""
Date and Time functions

"""

from datetime import timedelta, datetime

# ==============================
# Check if date is a Brazilian holiday
# ==============================
def is_br_holiday(dt: datetime) -> bool:
    """
    Check if a date is a Brazilian holiday
    """
    import holidays
    try:
        br = holidays.Brazil(years=dt.year)
        return dt in br
    except Exception:
        return False

# ==============================
# Check if it is a financial date
# ==============================
def get_next_financial_date(last_date: datetime) -> datetime:
    """
    Get next Brazilian commercial date
    """
    nd = last_date
    while True:
        nd += timedelta(days=1)
        if nd.weekday() == 6:  # sunday
            continue
        if is_br_holiday(nd):
            continue
        return nd