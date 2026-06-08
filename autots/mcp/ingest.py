"""Smart data ingestion for the AutoTS PWA / MCP server.

``smart_load`` is the beginner-friendly entry point for getting messy,
human-authored spreadsheets into a clean wide ``DataFrame`` ready for
forecasting. It is transport-neutral (no MCP / Pyodide dependency) so it is
shared by the MCP server, the Pyodide worker, and any general-purpose API.

The hardest part of the beginner flow is the upload: non-technical users export
spreadsheets that contain stray calculations, off-center tables, and empty
padding rows/columns. ``smart_load`` reproduces the cleanup described in the PWA
design doc:

  1. Parse pasted text (CSV/TSV), uploaded bytes (CSV/Excel), or a URL.
  2. Drop fully-empty rows/columns, then rows/columns that are >=95% empty.
  3. Auto-detect orientation (single-series, wide, or long) and return a clean
     wide ``DataFrame`` plus a human/LLM-readable report of what happened.
"""

import io
import warnings

import numpy as np
import pandas as pd

from autots.tools.shaping import long_to_wide, infer_frequency

# Column-name hints (lower-cased, exact match after strip) used to disambiguate
# roles when the data alone is ambiguous.
_DATE_HINTS = {"datetime", "date", "timestamp", "time", "ds", "period", "month"}
_VALUE_HINTS = {"value", "values", "y", "amount", "qty", "quantity", "sales"}
_ID_HINTS = {"series_id", "series", "id", "name", "label", "category", "item", "sku"}

# A row/column is discarded if its fraction of non-null cells is <= this value
# (i.e. it is >=95% empty).
_EMPTY_FRACTION = 0.05


def _read_raw(text=None, csv_bytes=None, url=None, filename=None):
    """Read any supported source into a raw header-less DataFrame (a grid).

    We read without assuming a header or index so the cleanup step can strip
    padding before we decide what the real header row is.
    """
    name = (filename or "").lower()

    if csv_bytes is not None and (name.endswith(".xlsx") or name.endswith(".xls")):
        return pd.read_excel(io.BytesIO(csv_bytes), header=None)

    if text is not None:
        buffer = io.StringIO(text)
    elif csv_bytes is not None:
        buffer = io.StringIO(csv_bytes.decode("utf-8", errors="replace"))
    elif url is not None:
        buffer = url
    else:
        raise ValueError("Must provide one of: text, csv_bytes, or url")

    # sep=None + engine='python' sniffs the delimiter (handles CSV and TSV).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_csv(
            buffer,
            sep=None,
            engine="python",
            header=None,
            skip_blank_lines=False,
        )


def _strip_empty(df):
    """Drop fully-empty then >=95%-empty rows and columns."""
    # Treat empty strings / whitespace as missing for the purpose of cleanup.
    mask = df.replace(r"^\s*$", np.nan, regex=True)

    df = df.loc[mask.notna().any(axis=1), mask.notna().any(axis=0)]
    mask = mask.loc[df.index, df.columns]

    df = df.loc[mask.notna().mean(axis=1) > _EMPTY_FRACTION]
    mask = mask.loc[df.index]
    df = df.loc[:, mask.notna().mean(axis=0) > _EMPTY_FRACTION]
    return df.reset_index(drop=True)


def _numeric_score(series):
    return pd.to_numeric(series, errors="coerce").notna().mean()


def _datetime_score(series):
    """Fraction of values that parse as dates, ignoring purely-numeric columns.

    Plain integers/floats parse as nanosecond timestamps, so we exclude columns
    that are overwhelmingly numeric to avoid mistaking an id/value for a date.
    """
    if _numeric_score(series) > 0.9:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed.notna().mean()


def _promote_header(df):
    """Decide whether the first remaining row is a header and apply it."""
    first = df.iloc[0]
    # Header rows are labels: mostly non-numeric and containing no real dates.
    # If the first row is mostly numeric, or contains any parseable date, it is
    # data rather than a header.
    looks_like_data = (_numeric_score(first) > 0.5) or (_datetime_score(first) > 0.3)
    if looks_like_data:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        return df, False
    header = [str(h).strip() for h in first]
    df = df.iloc[1:].copy()
    df.columns = header
    return df.reset_index(drop=True), True


def _classify_columns(df):
    """Return per-column role scores for date/value/id detection."""
    info = {}
    for col in df.columns:
        s = df[col]
        info[col] = {
            "date": _datetime_score(s),
            "numeric": _numeric_score(s),
            "name": str(col).strip().lower(),
        }
    return info


def _to_datetime_index(df, date_col):
    out = df.drop(columns=[date_col]).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.index = pd.to_datetime(df[date_col], errors="coerce")
    out = out[out.index.notna()]
    out.index.name = "datetime"
    # Coerce remaining columns to numeric where possible.
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_index()


def smart_load(
    *,
    text=None,
    csv_bytes=None,
    url=None,
    filename=None,
    data_format="auto",
    long_cols=None,
):
    """Load messy user data into a clean wide DataFrame.

    Args:
        text: Pasted CSV/TSV text.
        csv_bytes: Raw uploaded bytes (CSV or Excel; Excel detected via filename).
        url: A CSV URL (e.g. a published Google Sheet).
        filename: Optional original filename, used to detect Excel uploads.
        data_format: "auto" (default), "wide", or "long".
        long_cols: For long data, an optional dict
            ``{"date": ..., "value": ..., "id": ...}`` naming the columns. When
            omitted (or "auto"), columns are auto-detected.

    Returns:
        (df_wide, report) where ``df_wide`` has a DatetimeIndex and one column
        per series, and ``report`` is a JSON-serializable dict describing the
        cleanup and detection decisions.
    """
    raw = _read_raw(text=text, csv_bytes=csv_bytes, url=url, filename=filename)
    raw_shape = list(raw.shape)

    cleaned = _strip_empty(raw)
    if cleaned.empty:
        raise ValueError("No data remained after removing empty rows and columns")

    df, had_header = _promote_header(cleaned)
    info = _classify_columns(df)

    report = {
        "raw_shape": {"rows": raw_shape[0], "columns": raw_shape[1]},
        "cleaned_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dropped": {
            "rows": int(raw_shape[0] - df.shape[0] - (1 if had_header else 0)),
            "columns": int(raw_shape[1] - df.shape[1]),
        },
        "had_header": had_header,
    }

    # --- pick the datetime column (best score, with name-hint tie-break) ------
    def _date_key(col):
        return (info[col]["date"], info[col]["name"] in _DATE_HINTS)

    date_col = max(df.columns, key=_date_key)
    if info[date_col]["date"] < 0.5:
        raise ValueError(
            "Could not identify a date/datetime column. Ensure one column "
            "contains dates (e.g. 2024-01-31)."
        )

    other_cols = [c for c in df.columns if c != date_col]
    numeric_cols = [c for c in other_cols if info[c]["numeric"] > 0.8]
    string_cols = [c for c in other_cols if info[c]["numeric"] <= 0.8]

    # --- decide orientation ---------------------------------------------------
    fmt = data_format
    if fmt == "auto":
        if len(other_cols) == 1:
            fmt = "wide"  # single series
        elif string_cols and numeric_cols:
            fmt = "long"  # datetime + id(s) + value
        else:
            fmt = "wide"  # datetime + many numeric series

    if fmt == "long":
        lc = long_cols if isinstance(long_cols, dict) else {}
        id_col = lc.get("id") or (
            next((c for c in string_cols if info[c]["name"] in _ID_HINTS), None)
            or (string_cols[0] if string_cols else None)
        )
        value_col = lc.get("value") or (
            next((c for c in numeric_cols if info[c]["name"] in _VALUE_HINTS), None)
            or (numeric_cols[0] if numeric_cols else None)
        )
        date_col = lc.get("date", date_col)
        if id_col is None or value_col is None:
            raise ValueError(
                "Long format requires a series-id column and a value column."
            )
        long_df = df[[date_col, id_col, value_col]].copy()
        long_df[value_col] = pd.to_numeric(long_df[value_col], errors="coerce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_wide = long_to_wide(
                long_df,
                date_col=date_col,
                value_col=value_col,
                id_col=id_col,
                aggfunc="first",
            )
        report["detected_format"] = "long"
        report["long_columns"] = {
            "date": str(date_col),
            "id": str(id_col),
            "value": str(value_col),
        }
    else:
        df_wide = _to_datetime_index(df[[date_col] + other_cols], date_col)
        # Drop columns that are entirely non-numeric (couldn't be coerced).
        df_wide = df_wide.dropna(axis=1, how="all")
        report["detected_format"] = "wide"
        report["single_series"] = len(df_wide.columns) == 1

    df_wide = df_wide[~df_wide.index.duplicated(keep="first")].sort_index()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            freq = infer_frequency(df_wide, warn=False)
    except Exception:
        freq = None
    report["series_count"] = int(df_wide.shape[1])
    report["row_count"] = int(df_wide.shape[0])
    report["inferred_frequency"] = None if freq is None else str(freq)
    report["series_names"] = [str(c) for c in df_wide.columns]
    report["date_range"] = {
        "start": df_wide.index.min().strftime("%Y-%m-%d"),
        "end": df_wide.index.max().strftime("%Y-%m-%d"),
    } if len(df_wide) else None

    return df_wide, report
