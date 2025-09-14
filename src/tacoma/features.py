"""
Feature engineering helpers for the Tacoma load research homework.
"""
from __future__ import annotations
import pandas as pd
import pathlib

def _coerce_datetime(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    if datetime_col not in df.columns:
        # try case-insensitive match
        cols = {c.lower(): c for c in df.columns}
        key = datetime_col.lower()
        if key in cols:
            datetime_col = cols[key]
        else:
            raise KeyError(f"Datetime column '{datetime_col}' not found in columns: {list(df.columns)}")
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    if df[datetime_col].isna().any():
        df = df.dropna(subset=[datetime_col]).copy()
    return df, datetime_col

def load_and_prepare_data(
    path: str | pathlib.Path,
    datetime_col: str = "date time",
    usage_col: str = "kWh",
    temp_col: str | None = None,
    winter_months: tuple[int, ...] = (11, 12, 1, 2, 3),
    weekend_days: tuple[int, ...] = (5, 6),
    tz: str | None = None,
    set_index: bool = True,
) -> pd.DataFrame:
    """
    Load interval data and add season/day-type features commonly used in the analysis.

    Parameters
    ----------
    path : str | Path
        Path to Excel/CSV file.
    datetime_col : str
        Name of the datetime column (default: 'date time').
    usage_col : str
        Name of the energy usage column (e.g., 'kWh').
    temp_col : str | None
        Optional temperature column name, if available.
    winter_months : tuple[int, ...]
        Months considered "winter" (default: Nov–Mar).
    weekend_days : tuple[int, ...]
        Day-of-week values considered weekend (0=Mon .. 6=Sun). Default: (5,6).
    tz : str | None
        Optional IANA timezone to localize/convert to (e.g., 'America/Los_Angeles').
    set_index : bool
        If True, sets the datetime column as the index.

    Returns
    -------
    DataFrame with added columns:
    - 'date' (date component)
    - 'day_of_week' (0=Mon..6=Sun)
    - 'day_of_month', 'day_of_year'
    - 'hour_of_day', 'hour_of_week', 'hour_of_month', 'hour_of_year'
    - 'is_winter' (bool), 'is_weekend' (bool)
    - 'season' ('winter'|'non_winter')
    - 'day_type' ('weekday'|'weekend')
    - 'season_daytype' (e.g., 'winter_weekday')
    """
    path = pathlib.Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        # attempt excel then csv
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_csv(path)

    df, datetime_col = _coerce_datetime(df, datetime_col)

    # timezone handling
    if tz is not None:
        # localize if naive, otherwise convert
        if df[datetime_col].dt.tz is None:
            df[datetime_col] = df[datetime_col].dt.tz_localize(tz)
        else:
            df[datetime_col] = df[datetime_col].dt.tz_convert(tz)

    # index option
    if set_index:
        df = df.set_index(datetime_col).sort_index()
        idx = df.index
        df["date"] = idx.date
        df["day_of_week"] = idx.dayofweek
        df["day_of_month"] = idx.day
        df["day_of_year"] = idx.dayofyear
        df["hour_of_day"] = idx.hour
        df["hour_of_week"] = (idx.dayofweek * 24) + idx.hour
        df["hour_of_month"] = ((idx.day - 1) * 24) + idx.hour
        df["hour_of_year"] = ((idx.dayofyear - 1) * 24) + idx.hour
        df["month"] = idx.month
    else:
        dt = df[datetime_col].dt
        df["date"] = dt.date
        df["day_of_week"] = dt.dayofweek
        df["day_of_month"] = dt.day
        df["day_of_year"] = dt.dayofyear
        df["hour_of_day"] = dt.hour
        df["hour_of_week"] = (dt.dayofweek * 24) + dt.hour
        df["hour_of_month"] = ((dt.day - 1) * 24) + dt.hour
        df["hour_of_year"] = ((dt.dayofyear - 1) * 24) + dt.hour
        df["month"] = dt.month

    # Season / weekend flags
    df["is_winter"] = df["month"].isin(winter_months)
    df["is_weekend"] = df["day_of_week"].isin(weekend_days)
    df["season"] = df["is_winter"].map({True: "winter", False: "non_winter"})
    df["day_type"] = df["is_weekend"].map({True: "weekend", False: "weekday"})
    df["season_daytype"] = df["season"] + "_" + df["day_type"]

    # Ensure usage column exists (case-insensitive fallback)
    if usage_col not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        if usage_col.lower() in cols:
            usage_col = cols[usage_col.lower()]
        else:
            # leave as-is for downstream
            pass

    # Optional temp col normalization
    if temp_col is not None and temp_col not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        if temp_col.lower() in cols:
            temp_col = cols[temp_col.lower()]

    return df
