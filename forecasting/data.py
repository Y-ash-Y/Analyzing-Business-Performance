"""Construct the raw Friday panel and prepare training data at a cutoff."""

from pathlib import Path

import numpy as np
import pandas as pd

FREQUENCY = "W-FRI"
COMPANIES = {1: "Bigg Bazar", 2: "D-Mart", 3: "Spar", 4: "Seven Eleven"}
KEYS = ["company", "Store", "Dept"]
COLUMNS = ["Date", "Store", "Dept", "Weekly_Sales"]
# Fixed evaluation scope from the source coverage audit, with no sales-based selection.
INCOMPLETE_SERIES = [("Seven Eleven", 3, 5), ("Seven Eleven", 3, 6),
                     ("Seven Eleven", 4, 6), ("Seven Eleven", 5, 6)]


def load_panel(directory: Path) -> pd.DataFrame:
    """Return one column per expected series, leaving missing sales as NaN."""
    parts = []
    for number, company in COMPANIES.items():
        frame = pd.read_csv(directory / f"company{number}.csv", usecols=COLUMNS)
        frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
        if frame[["Date", "Store", "Dept"]].isna().any().any():
            raise ValueError(f"{company} has missing key values")
        if (frame["Date"].dt.dayofweek.ne(4) |
                frame["Date"].ne(frame["Date"].dt.normalize())).any():
            raise ValueError(f"{company} has dates outside the Friday calendar")
        if not frame["Store"].isin(range(1, 8)).all():
            raise ValueError(f"{company} has unexpected store identifiers")
        if number == 3:
            frame = frame.loc[frame["Dept"].isin(range(1, 7))].copy()
        elif not frame["Dept"].isin(range(1, 7)).all():
            raise ValueError(f"{company} has unexpected department identifiers")
        frame["Weekly_Sales"] = pd.to_numeric(frame["Weekly_Sales"], errors="raise")
        if np.isinf(frame["Weekly_Sales"]).any():
            raise ValueError(f"{company} has infinite sales values")
        parts.append(frame.assign(company=company))

    rows = pd.concat(parts, ignore_index=True)
    if rows.duplicated(["Date", *KEYS]).any():
        raise ValueError("Duplicate company/store/department/Friday keys")
    panel = rows.pivot(index="Date", columns=KEYS, values="Weekly_Sales")
    calendar = pd.date_range(panel.index.min(), panel.index.max(), freq=FREQUENCY, name="Date")
    series = pd.MultiIndex.from_product(
        [COMPANIES.values(), range(1, 8), range(1, 7)], names=KEYS
    )
    if not series.isin(panel.columns).all():
        raise ValueError("One or more expected store-department series have no source rows")
    return panel.reindex(index=calendar, columns=series)


def missing_gaps(panel: pd.DataFrame) -> pd.DataFrame:
    """List contiguous missing runs, including leading and trailing gaps."""
    rows = []
    for key, series in panel.items():
        missing = series.isna()
        starts = series.index[missing & ~missing.shift(fill_value=False)]
        ends = series.index[missing & ~missing.shift(-1, fill_value=False)]
        rows.extend((*key, start, end, (end - start).days // 7 + 1)
                    for start, end in zip(starts, ends, strict=True))
    return pd.DataFrame(rows, columns=[*KEYS, "start", "end", "weeks"])


def complete_scope(panel: pd.DataFrame) -> pd.DataFrame:
    """Select the predeclared 164-series benchmark and verify its coverage."""
    complete = panel.drop(columns=INCOMPLETE_SERIES)
    if complete.isna().any().any():
        raise ValueError("Missing sales in the fixed 164-series benchmark cohort")
    return complete


def prepare_training(panel: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.Series, pd.DataFrame]:
    """Fill training gaps using only rows dated on or before each missing Friday."""
    if cutoff not in panel.index:
        raise ValueError("Training cutoff must be a source calendar Friday")
    observed = panel.loc[:cutoff]
    carried = observed.ffill()
    prepared = carried.copy()
    audit = []
    for key in observed.columns[observed.isna().any()]:
        original = observed[key]
        missing = original.isna()
        leading = missing & carried[key].isna()
        if leading.any():
            company, _, department = key
            peers = observed.loc[:, (company, slice(None), department)].drop(columns=[key])
            prepared.loc[leading, key] = peers.loc[leading].median(axis=1)
        for date in original.index[missing]:
            method = "same_week_peer_median" if leading.loc[date] else "carry_forward"
            audit.append((*key, date, method, prepared.at[date, key]))
    if prepared.isna().any().any():
        raise ValueError("Training gap has neither a prior observation nor a same-Friday peer")
    fills = pd.DataFrame(audit, columns=[*KEYS, "Date", "method", "imputed_sales"])
    return prepared.sum(axis=1).rename("sales"), fills


def observed_totals(panel: pd.DataFrame) -> pd.Series:
    """Aggregate only weeks where every series in the scope has observed sales."""
    return panel.sum(axis=1, min_count=panel.shape[1]).rename("actual")
