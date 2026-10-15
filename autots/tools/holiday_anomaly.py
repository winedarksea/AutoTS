"""Per-series holiday anomaly suppression with comparable annual peers."""

import numpy as np
import pandas as pd


def _holiday_occurrences(index, holiday_groups, proximity_days):
    """Assign each calendar day its closest, best-supported holiday family."""
    frames = []
    anchors_by_family = {}
    for family, dates in holiday_groups.items():
        anchors = []
        for date in dates:
            date = pd.Timestamp(date)
            if index.tz is not None:
                date = (
                    date.tz_localize(index.tz)
                    if date.tz is None
                    else date.tz_convert(index.tz)
                )
            elif date.tz is not None:
                date = date.tz_localize(None)
            if pd.notna(date):
                anchors.append(date.normalize())
        anchors = pd.DatetimeIndex(anchors).unique().sort_values()
        if anchors.empty:
            continue
        anchors_by_family[family] = anchors
        support = len(anchors.year.unique())
        for offset in range(-proximity_days, proximity_days + 1):
            frames.append(
                pd.DataFrame(
                    {
                        "date": anchors + pd.DateOffset(days=offset),
                        "family": family,
                        "offset": offset,
                        "distance": abs(offset),
                        "support": support,
                        "year": anchors.year,
                    }
                )
            )
    if not frames:
        return None, anchors_by_family
    occurrences = (
        pd.concat(frames, ignore_index=True)
        .sort_values(
            ["distance", "support", "family", "date"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates("date")
        .set_index("date")
    )
    return occurrences, anchors_by_family


def named_holiday_suppression_mask(
    anomalies,
    strengths,
    observed,
    holiday_dates,
    proximity_days,
    override,
    shared=False,
    residuals=None,
):
    """Return suppression eligibility; unknown peers conservatively retain flags.

    Each comparison excludes the candidate's year. Intraday observations compare
    the same local time, and missing source observations never count as peers.
    """
    index = anomalies.index
    per_series = pd.DataFrame(False, index=index, columns=observed.columns)
    for column in observed.columns:
        groups = holiday_dates.get(column, {})
        occurrences, anchors_by_family = _holiday_occurrences(
            index, groups, max(0, int(proximity_days))
        )
        if occurrences is None:
            continue
        assigned = occurrences.reindex(index.normalize())
        assigned.index = index
        near = assigned["family"].notna()
        if override is None:
            per_series[column] = near
            continue
        if strengths is None:
            continue
        strength = strengths.iloc[:, 0] if shared else strengths.get(column)
        if strength is None:
            continue
        strength = strength.reindex(index).where(observed[column])
        magnitude = None
        if residuals is not None and column in residuals:
            residual = residuals[column].reindex(index).where(observed[column])
            magnitude = (residual - residual.median()).abs()
        flagged = anomalies.iloc[:, 0].eq(-1) if shared else anomalies[column].eq(-1)
        candidates = assigned.loc[near & flagged & strength.notna()].copy()
        candidates["time"] = candidates.index - candidates.index.normalize()
        for (family, offset, time), group in candidates.groupby(
            ["family", "offset", "time"]
        ):
            peer_dates = (
                anchors_by_family[family] + pd.DateOffset(days=int(offset)) + time
            )
            peer_strengths = strength.reindex(peer_dates)
            # A family can contain several occurrences in a year; it still gets
            # only one vote, rather than manufacturing additional peer years.
            annual_peers = peer_strengths.groupby(
                anchors_by_family[family].year
            ).median()
            annual_peers = annual_peers[np.isfinite(annual_peers)]
            annual_magnitudes = None
            if magnitude is not None:
                annual_magnitudes = (
                    magnitude.reindex(peer_dates)
                    .groupby(anchors_by_family[family].year)
                    .median()
                )
            for year in group["year"].unique():
                peers = annual_peers.loc[annual_peers.index != year]
                if len(peers) < 2:
                    continue
                dates = group.index[group["year"] == year]
                values = strength.loc[dates]
                suppress = np.isfinite(values) & values.lt(
                    float(override) * peers.median()
                )
                if annual_magnitudes is not None:
                    magnitude_peers = annual_magnitudes.loc[
                        annual_magnitudes.index != year
                    ]
                    magnitude_peers = magnitude_peers[np.isfinite(magnitude_peers)]
                    reference = magnitude_peers.median()
                    # Rolling scores normalize each isolated spike by its own
                    # variance, hiding size differences between annual events.
                    if len(magnitude_peers) >= 2 and reference > 0:
                        suppress &= ~magnitude.loc[dates].ge(
                            float(override) * reference
                        )
                per_series.loc[dates, column] = suppress.to_numpy()
    if shared:
        # One shared flag cannot be cleared on behalf of only some columns.
        return pd.DataFrame(
            np.repeat(
                per_series.all(axis=1).to_numpy()[:, None], anomalies.shape[1], axis=1
            ),
            index=index,
            columns=anomalies.columns,
        )
    return per_series.reindex(columns=anomalies.columns, fill_value=False)
