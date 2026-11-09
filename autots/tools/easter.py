"""Western Easter offsets and data-driven recurring holiday rules."""

import pandas as pd

try:
    from dateutil.easter import easter
except ImportError:
    easter = None


def easter_offsets(dates):
    """Return calendar-day offsets from Western Easter in each date's year."""
    if easter is None:
        raise ImportError(
            "Easter features require python-dateutil (a pandas dependency)."
        )
    dates = pd.DatetimeIndex(dates)
    local_dates = dates.tz_localize(None).normalize()
    easter_by_year = {
        year: pd.Timestamp(easter(int(year))) for year in dates.year.unique()
    }
    anchors = pd.DatetimeIndex([easter_by_year[year] for year in dates.year])
    return pd.Series((local_dates - anchors).days, index=dates, name="easter_offset")


def extract_easter_holidays(
    anomalies, threshold, min_occurrences, actuals=None, anomaly_scores=None
):
    """Select supported offsets independently; a calendar date alone is not a rule."""
    if isinstance(anomalies, pd.Series):
        anomalies = anomalies.to_frame(name="all")
    offsets = easter_offsets(anomalies.index)
    eligible = offsets.between(-2, 1)
    observed = anomalies.loc[eligible]
    flagged = observed.eq(-1).astype(float).where(observed.notna())
    stacked = (
        flagged.rename_axis(index="date", columns="series")
        .stack()
        .rename("count")
        .to_frame()
    )
    stacked["occurrence_rate"] = stacked["count"]
    aggregation = {"count": "sum", "occurrence_rate": "mean"}
    for frame, name in [(actuals, "avg_value"), (anomaly_scores, "avg_anomaly_score")]:
        if frame is not None:
            values = frame.reindex(index=observed.index, columns=observed.columns)
            stacked[name] = values.rename_axis(index="date", columns="series").stack()
            aggregation[name] = "mean"
    stacked["easter_offset"] = offsets.reindex(
        stacked.index.get_level_values("date")
    ).to_numpy()
    rules = stacked.groupby(["series", "easter_offset"]).agg(aggregation).reset_index()
    rules = rules.loc[
        (rules["occurrence_rate"] >= threshold) & (rules["count"] >= min_occurrences)
    ].copy()
    rules["holiday_name"] = rules["easter_offset"].map(
        {-2: "easter_-2", -1: "easter_-1", 0: "easter_0", 1: "easter_+1"}
    )
    return rules
