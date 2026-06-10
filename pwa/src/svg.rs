//! Minimal dependency-free SVG line chart.
//!
//! Rendered as an inline SVG string. Every chart is paired with an accessible
//! DOM table elsewhere, so the SVG is decorative (aria-hidden) — screen readers
//! and LLMs read the structured table instead.
//!
//! Interaction (hover tooltip, crosshair, drag-to-zoom) is NOT baked into the
//! SVG: the host (`app.rs`) overlays signal-driven elements and hit-tests the
//! cursor through the [`ChartGeom`] returned alongside the SVG string, so the
//! SVG can stay a static string injected via `inner_html`.

use crate::dates::{fmt_tick, Granularity};

pub const VB_W: f64 = 920.0;
pub const VB_H: f64 = 340.0;
const W: f64 = VB_W;
const H: f64 = VB_H;
const PAD_L: f64 = 48.0;
const PAD_R: f64 = 16.0;
const PAD_T: f64 = 16.0;
const PAD_B: f64 = 38.0; // extra room for x-axis date labels

/// One line to draw: history portion (solid) + optional forecast portion (dashed),
/// optional uncertainty band — all in the same palette color. `color_idx` maps to
/// `--viz-s{n}` (wraps at 7). History and forecast slices align to a shared axis:
/// history occupies absolute indices `0..H`, forecast `H..H+F`.
pub struct ChartSeries<'a> {
    pub h: &'a [f64],
    pub f: &'a [f64],
    pub upper: Option<&'a [f64]>,
    pub lower: Option<&'a [f64]>,
    pub color_idx: usize,
}

impl ChartSeries<'_> {
    fn color(&self) -> String {
        format!("var(--viz-s{})", self.color_idx % 7)
    }
}

/// A detected-feature glyph to draw at one data point. `idx` is an absolute
/// (history) axis index; `value` is the series value there.
#[derive(Clone, Copy)]
pub struct FeatureMarker {
    pub idx: usize,
    pub kind: FeatureKind,
    pub value: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FeatureKind {
    Anomaly,
    LevelShift,
    Changepoint,
    Holiday,
}

impl FeatureKind {
    pub const ALL: [FeatureKind; 4] = [
        FeatureKind::Anomaly,
        FeatureKind::LevelShift,
        FeatureKind::Changepoint,
        FeatureKind::Holiday,
    ];

    pub fn index(self) -> usize {
        match self {
            FeatureKind::Anomaly => 0,
            FeatureKind::LevelShift => 1,
            FeatureKind::Changepoint => 2,
            FeatureKind::Holiday => 3,
        }
    }

    /// JSON key in `features["series"][name]` (the per-feature date lists).
    pub fn json_key(self) -> &'static str {
        match self {
            FeatureKind::Anomaly => "anomalies",
            FeatureKind::LevelShift => "level_shifts",
            FeatureKind::Changepoint => "trend_changepoints",
            FeatureKind::Holiday => "holiday_dates",
        }
    }

    /// Key in the `detection_counts` summary (note: holidays differs from json_key).
    pub fn count_key(self) -> &'static str {
        match self {
            FeatureKind::Anomaly => "anomalies",
            FeatureKind::LevelShift => "level_shifts",
            FeatureKind::Changepoint => "trend_changepoints",
            FeatureKind::Holiday => "holidays",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            FeatureKind::Anomaly => "anomalies",
            FeatureKind::LevelShift => "level shifts",
            FeatureKind::Changepoint => "trend changes",
            FeatureKind::Holiday => "holidays",
        }
    }

    pub fn tooltip_label(self) -> &'static str {
        match self {
            FeatureKind::Anomaly => "ANOMALY",
            FeatureKind::LevelShift => "LEVEL SHIFT",
            FeatureKind::Changepoint => "TREND CHANGE",
            FeatureKind::Holiday => "HOLIDAY",
        }
    }

    /// Fixed per-kind color so the marker type is legible across series.
    fn color(self) -> &'static str {
        match self {
            FeatureKind::Anomaly => "var(--md-error)",
            FeatureKind::LevelShift => "var(--md-tertiary)",
            FeatureKind::Changepoint => "var(--md-secondary)",
            FeatureKind::Holiday => "var(--md-primary)",
        }
    }
}

/// Which feature kinds are currently shown (one bool per [`FeatureKind`]).
#[derive(Clone, Copy)]
pub struct FeatureKindSet(pub [bool; 4]);

impl FeatureKindSet {
    pub fn none() -> Self {
        FeatureKindSet([false; 4])
    }
    pub fn get(self, k: FeatureKind) -> bool {
        self.0[k.index()]
    }
    pub fn with_toggled(mut self, k: FeatureKind) -> Self {
        self.0[k.index()] = !self.0[k.index()];
        self
    }
}

/// Geometry of a rendered chart, shared between drawing and pointer hit-testing.
/// Coordinates are in the fixed 920×340 viewBox; `base_idx` is the absolute axis
/// index of local position 0 (non-zero when zoomed).
#[derive(Clone, Copy)]
pub struct ChartGeom {
    pub base_idx: usize,
    pub n_local: usize,
    pub lo: f64,
    pub hi: f64,
    pub xstep: f64,
}

impl ChartGeom {
    fn plot_h() -> f64 {
        H - PAD_T - PAD_B
    }
    pub fn vbx_of_abs(&self, abs: usize) -> f64 {
        PAD_L + (abs as f64 - self.base_idx as f64) * self.xstep
    }
    pub fn y_of(&self, v: f64) -> f64 {
        if (self.hi - self.lo).abs() < f64::EPSILON {
            return PAD_T + Self::plot_h() * 0.5;
        }
        PAD_T + Self::plot_h() * (1.0 - (v - self.lo) / (self.hi - self.lo))
    }
    /// Nearest absolute data index for a viewBox-space x coordinate.
    pub fn index_at_vbx(&self, vbx: f64) -> usize {
        if self.n_local == 0 {
            return self.base_idx;
        }
        let local = ((vbx - PAD_L) / self.xstep).round();
        let local = local.clamp(0.0, (self.n_local - 1) as f64) as usize;
        self.base_idx + local
    }
}

pub struct ChartOutput {
    pub svg: String,
    pub geom: ChartGeom,
}

fn finite_min_max(slices: &[&[f64]]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for s in slices {
        for &v in *s {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    if lo.is_finite() && hi.is_finite() {
        if (hi - lo).abs() < f64::EPSILON {
            return Some((lo - 1.0, hi + 1.0));
        }
        Some((lo, hi))
    } else {
        None
    }
}

/// Compact numeric label that fits in the y-axis gutter / tooltip box.
pub fn fmt_value(v: f64) -> String {
    let abs = v.abs();
    if abs >= 1_000_000.0 {
        format!("{:.2}M", v / 1_000_000.0)
    } else if abs >= 10_000.0 {
        format!("{:.0}", v)
    } else if abs >= 1_000.0 {
        format!("{:.1}", v)
    } else if abs >= 10.0 {
        format!("{:.2}", v)
    } else {
        format!("{:.4}", v)
    }
}

/// Max local-x distance between consecutive finite points that we bridge with a
/// connecting line (≈ up to 3 missing points). Larger gaps break the pen so a
/// genuinely-absent stretch isn't drawn as a misleading flat segment.
const MAX_GAP_BRIDGE: f64 = 4.0;

/// Build an SVG path "M…L…" from (local-x-index, value) points. Short runs of
/// non-finite values are bridged with a connecting line (so the line doesn't
/// fragment when zoomed); gaps wider than [`MAX_GAP_BRIDGE`] break the pen.
fn path_from(points: &[(f64, f64)], geom: &ChartGeom) -> String {
    let mut d = String::new();
    let mut prev_lx: Option<f64> = None;
    for &(lx, v) in points {
        if !v.is_finite() {
            continue; // skip missing data; decide below whether to bridge it
        }
        let x = PAD_L + lx * geom.xstep;
        let y = geom.y_of(v);
        let bridge = matches!(prev_lx, Some(p) if lx - p <= MAX_GAP_BRIDGE);
        if bridge {
            d.push_str(&format!(" L{x:.1} {y:.1}"));
        } else {
            d.push_str(&format!(" M{x:.1} {y:.1}"));
        }
        prev_lx = Some(lx);
    }
    d
}

/// Filled uncertainty-band polygon between `upper` and `lower` over the same
/// local-x points (forward along upper, back along lower). Only emitted over the
/// maximal contiguous finite run.
fn band_path(points_upper: &[(f64, f64)], points_lower: &[(f64, f64)], geom: &ChartGeom) -> String {
    // Collect the contiguous prefix where both bounds are finite.
    let mut up = Vec::new();
    let mut lo = Vec::new();
    for (&(lx, u), &(_, l)) in points_upper.iter().zip(points_lower.iter()) {
        if u.is_finite() && l.is_finite() {
            up.push((lx, u));
            lo.push((lx, l));
        } else if !up.is_empty() {
            break;
        }
    }
    if up.len() < 2 {
        return String::new();
    }
    let mut d = String::new();
    for (i, &(lx, u)) in up.iter().enumerate() {
        let x = PAD_L + lx * geom.xstep;
        let y = geom.y_of(u);
        d.push_str(&format!("{}{x:.1} {y:.1}", if i == 0 { "M" } else { " L" }));
    }
    for &(lx, l) in lo.iter().rev() {
        let x = PAD_L + lx * geom.xstep;
        let y = geom.y_of(l);
        d.push_str(&format!(" L{x:.1} {y:.1}"));
    }
    d.push_str(" Z");
    d
}

/// A glyph marking one detected feature at viewBox (x, y).
fn marker_shape(kind: FeatureKind, x: f64, y: f64) -> String {
    let c = kind.color();
    let r = 4.0;
    let stroke = "stroke=\"var(--md-surface)\" stroke-width=\"1\"";
    match kind {
        FeatureKind::Anomaly => format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
             fill=\"{c}\" {stroke}/>",
            x - r,
            y - r,
            r * 2.0,
            r * 2.0
        ),
        FeatureKind::LevelShift => format!(
            "<path d=\"M{x:.1} {:.1} L{:.1} {:.1} L{:.1} {:.1} Z\" fill=\"{c}\" {stroke}/>",
            y - r,
            x + r,
            y + r,
            x - r,
            y + r
        ),
        FeatureKind::Changepoint => format!(
            "<path d=\"M{x:.1} {:.1} L{:.1} {y:.1} L{x:.1} {:.1} L{:.1} {y:.1} Z\" \
             fill=\"{c}\" {stroke}/>",
            y - r,
            x + r,
            y + r,
            x - r
        ),
        FeatureKind::Holiday => {
            format!("<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{r:.1}\" fill=\"{c}\" {stroke}/>")
        }
    }
}

/// A standalone 14×14 SVG of one feature kind's glyph, for the interactive legend.
pub fn legend_glyph(kind: FeatureKind) -> String {
    format!(
        "<svg viewBox=\"0 0 14 14\" width=\"13\" height=\"13\" \
         xmlns=\"http://www.w3.org/2000/svg\" aria-hidden=\"true\">{}</svg>",
        marker_shape(kind, 7.0, 7.0)
    )
}

/// Build an SVG for one or more series sharing a common date axis.
///
/// - `view_range` (absolute, inclusive) zooms to a window; `None` = full fit.
/// - History portions are solid lines; forecast portions dashed — same color.
/// - `show_bands` fills the upper/lower uncertainty band behind the lines.
/// - `markers` (per visible series, same order as `series`) draws detected-feature
///   glyphs for the kinds enabled in `enabled`.
/// - The y-axis auto-scales to the combined range of everything visible.
#[allow(clippy::too_many_arguments)]
pub fn line_chart_ex(
    series: &[ChartSeries<'_>],
    history_dates: &[String],
    forecast_dates: &[String],
    view_range: Option<(usize, usize)>,
    show_bands: bool,
    markers: &[Vec<FeatureMarker>],
    enabled: FeatureKindSet,
    gran: Granularity,
) -> ChartOutput {
    let empty_geom = ChartGeom {
        base_idx: 0,
        n_local: 0,
        lo: 0.0,
        hi: 1.0,
        xstep: 0.0,
    };
    if series.is_empty() {
        return ChartOutput {
            svg: String::new(),
            geom: empty_geom,
        };
    }

    let n_history = series.iter().map(|s| s.h.len()).max().unwrap_or(0);
    let n_forecast = series.iter().map(|s| s.f.len()).max().unwrap_or(0);
    let n_full = n_history + n_forecast;
    if n_full == 0 {
        return ChartOutput {
            svg: String::new(),
            geom: empty_geom,
        };
    }

    // Resolve the visible absolute window [a, b].
    let (a, b) = view_range
        .map(|(s, e)| (s.min(n_full - 1), e.min(n_full - 1)))
        .map(|(s, e)| if s <= e { (s, e) } else { (e, s) })
        .unwrap_or((0, n_full - 1));
    let n_local = b - a + 1;

    // Per-series windowed point lists (local-x, value).
    struct Win {
        hist: Vec<(f64, f64)>,
        fc: Vec<(f64, f64)>,
        up: Vec<(f64, f64)>,
        lo: Vec<(f64, f64)>,
        color: String,
    }
    let mut wins: Vec<Win> = Vec::with_capacity(series.len());
    for s in series {
        let mut hist = Vec::new();
        for i in a..=b.min(n_history.saturating_sub(1)) {
            if i < n_history && i < s.h.len() {
                hist.push(((i - a) as f64, s.h[i]));
            }
        }
        let mut fc = Vec::new();
        let mut up = Vec::new();
        let mut lo = Vec::new();
        if n_forecast > 0 && b >= n_history {
            // connector from the last history point if the boundary is in view
            if n_history > 0 && a < n_history {
                if let Some(&hv) = s.h.get(n_history - 1) {
                    fc.push(((n_history - 1 - a) as f64, hv));
                }
            }
            let j_lo = a.saturating_sub(n_history);
            let j_hi = b - n_history;
            for j in j_lo..=j_hi {
                let lx = (n_history + j - a) as f64;
                if let Some(&fv) = s.f.get(j) {
                    fc.push((lx, fv));
                }
                if show_bands {
                    if let Some(u) = s.upper.and_then(|u| u.get(j)) {
                        up.push((lx, *u));
                    }
                    if let Some(l) = s.lower.and_then(|l| l.get(j)) {
                        lo.push((lx, *l));
                    }
                }
            }
        }
        wins.push(Win {
            hist,
            fc,
            up,
            lo,
            color: s.color(),
        });
    }

    // Y scale over every visible value (incl. bands when shown).
    let mut scale_slices: Vec<Vec<f64>> = Vec::new();
    for w in &wins {
        scale_slices.push(w.hist.iter().map(|p| p.1).collect());
        scale_slices.push(w.fc.iter().map(|p| p.1).collect());
        if show_bands {
            scale_slices.push(w.up.iter().map(|p| p.1).collect());
            scale_slices.push(w.lo.iter().map(|p| p.1).collect());
        }
    }
    let refs: Vec<&[f64]> = scale_slices.iter().map(|v| v.as_slice()).collect();
    // Preserve the chart frame and date axis for an all-missing selection.
    // An empty SVG makes the whole data card appear broken and hides the fact
    // that the chosen series simply has no observations in this window.
    let (lo_v, hi_v) = finite_min_max(&refs).unwrap_or((0.0, 1.0));

    let plot_w = W - PAD_L - PAD_R;
    let xstep = if n_local > 1 {
        plot_w / (n_local as f64 - 1.0)
    } else {
        plot_w
    };
    let geom = ChartGeom {
        base_idx: a,
        n_local,
        lo: lo_v,
        hi: hi_v,
        xstep,
    };

    let mut svg = format!(
        "<svg viewBox=\"0 0 {W} {H}\" role=\"img\" aria-hidden=\"true\" \
         xmlns=\"http://www.w3.org/2000/svg\">"
    );

    // y gridlines + labels (5 ticks)
    for t in 0..=4 {
        let frac = t as f64 / 4.0;
        let y = PAD_T + (H - PAD_T - PAD_B) * frac;
        let val = hi_v - (hi_v - lo_v) * frac;
        svg.push_str(&format!(
            "<line x1=\"{PAD_L:.0}\" y1=\"{y:.1}\" x2=\"{:.0}\" y2=\"{y:.1}\" \
             stroke=\"var(--md-outline-variant)\" stroke-width=\"1\"/>",
            W - PAD_R
        ));
        svg.push_str(&format!(
            "<text x=\"4\" y=\"{:.1}\" font-size=\"11\" \
             fill=\"var(--md-on-surface-variant)\">{}</text>",
            y + 4.0,
            fmt_value(val),
        ));
    }

    // x-axis date labels — up to 5 minimal ticks over the visible window
    let axis_y = H - PAD_B;
    let all_dates: Vec<&str> = history_dates
        .iter()
        .chain(forecast_dates.iter())
        .map(|s| s.as_str())
        .collect();
    if n_local > 0 {
        let k = 5usize;
        let ticks: Vec<usize> = if n_local <= k {
            (0..n_local).collect()
        } else {
            (0..k).map(|i| i * (n_local - 1) / (k - 1)).collect()
        };
        for local in ticks {
            let abs = a + local;
            let Some(&date_str) = all_dates.get(abs) else {
                continue;
            };
            if date_str.is_empty() {
                continue;
            }
            let x = PAD_L + local as f64 * xstep;
            let lx = x.clamp(PAD_L + 26.0, W - PAD_R - 26.0);
            let label = fmt_tick(date_str, gran);
            svg.push_str(&format!(
                "<line x1=\"{x:.1}\" y1=\"{axis_y:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" \
                 stroke=\"var(--md-outline-variant)\" stroke-width=\"1\"/>",
                axis_y + 4.0
            ));
            svg.push_str(&format!(
                "<text x=\"{lx:.1}\" y=\"{:.1}\" font-size=\"11\" \
                 fill=\"var(--md-on-surface-variant)\" text-anchor=\"middle\">{label}</text>",
                axis_y + 18.0,
            ));
        }
    }

    // Uncertainty bands first (behind everything).
    if show_bands {
        for w in &wins {
            if w.up.len() >= 2 && w.lo.len() >= 2 {
                let d = band_path(&w.up, &w.lo, &geom);
                if !d.is_empty() {
                    svg.push_str(&format!(
                        "<path d=\"{d}\" fill=\"var(--viz-uncertainty-fill)\" stroke=\"none\"/>"
                    ));
                    let upper_path = path_from(&w.up, &geom);
                    let lower_path = path_from(&w.lo, &geom);
                    svg.push_str(&format!(
                        "<path d=\"{upper_path}\" fill=\"none\" \
                         stroke=\"var(--viz-uncertainty-line)\" stroke-width=\"1.25\"/>"
                    ));
                    svg.push_str(&format!(
                        "<path d=\"{lower_path}\" fill=\"none\" \
                         stroke=\"var(--viz-uncertainty-line)\" stroke-width=\"1.25\"/>"
                    ));
                }
            }
        }
    }

    // history/forecast boundary marker (once, if the boundary is in view)
    if n_history > 0 && n_forecast > 0 && a < n_history && b >= n_history {
        let xb = PAD_L + (n_history as f64 - 0.5 - a as f64) * xstep;
        svg.push_str(&format!(
            "<line x1=\"{xb:.1}\" y1=\"{PAD_T:.0}\" x2=\"{xb:.1}\" y2=\"{axis_y:.1}\" \
             stroke=\"var(--md-outline)\" stroke-dasharray=\"4 4\" stroke-width=\"1\"/>",
        ));
    }

    // Lines (history solid, forecast dashed — same color each).
    for w in &wins {
        if !w.hist.is_empty() {
            let d = path_from(&w.hist, &geom);
            svg.push_str(&format!(
                "<path d=\"{d}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" \
                 stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
                w.color
            ));
        }
        if !w.fc.is_empty() {
            let d = path_from(&w.fc, &geom);
            svg.push_str(&format!(
                "<path d=\"{d}\" fill=\"none\" stroke=\"{}\" \
                 stroke-width=\"2.5\" stroke-dasharray=\"7 3\" \
                 stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
                w.color
            ));
        }
    }

    // Detected-feature markers (on top of the lines).
    for (si, marks) in markers.iter().enumerate() {
        let _ = si;
        for m in marks {
            if !enabled.get(m.kind) || !m.value.is_finite() {
                continue;
            }
            if m.idx < a || m.idx > b {
                continue;
            }
            let x = PAD_L + (m.idx - a) as f64 * xstep;
            let y = geom.y_of(m.value);
            svg.push_str(&marker_shape(m.kind, x, y));
        }
    }

    svg.push_str("</svg>");
    ChartOutput { svg, geom }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dates::Granularity;

    fn s(h: &'static [f64], f: &'static [f64]) -> ChartSeries<'static> {
        ChartSeries {
            h,
            f,
            upper: None,
            lower: None,
            color_idx: 0,
        }
    }

    fn plain(series: &[ChartSeries<'_>], hd: &[String], fd: &[String]) -> ChartOutput {
        line_chart_ex(
            series,
            hd,
            fd,
            None,
            false,
            &[],
            FeatureKindSet::none(),
            Granularity::Day,
        )
    }

    #[test]
    fn chart_distinguishes_history_and_forecast() {
        let out = plain(&[s(&[1.0, 2.0], &[3.0, 4.0])], &[], &[]);
        assert!(out.svg.contains("stroke-dasharray=\"7 3\"")); // dashed forecast
        assert!(out.svg.contains("stroke-linecap=\"round\""));
        assert!(out.svg.contains("stroke-dasharray=\"4 4\"")); // boundary marker
    }

    #[test]
    fn short_gap_is_bridged_into_one_subpath() {
        // A single interior NaN must not fragment the line: one move command.
        let out = plain(&[s(&[1.0, f64::NAN, 3.0], &[])], &[], &[]);
        assert_eq!(out.svg.matches(" M").count(), 1);
    }

    #[test]
    fn long_gap_breaks_the_pen() {
        // A run of NaNs wider than MAX_GAP_BRIDGE breaks into two subpaths.
        let out = plain(
            &[s(&[1.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, 6.0], &[])],
            &[],
            &[],
        );
        assert_eq!(out.svg.matches(" M").count(), 2);
    }

    #[test]
    fn multi_series_renders_both_colors() {
        let a = s(&[1.0, 2.0], &[]);
        let mut b = s(&[3.0, 4.0], &[]);
        b.color_idx = 1;
        let out = plain(&[a, b], &[], &[]);
        assert!(out.svg.contains("var(--viz-s0)"));
        assert!(out.svg.contains("var(--viz-s1)"));
    }

    #[test]
    fn all_missing_series_keeps_chart_frame_visible() {
        let dates = vec!["2024-01-01".to_string(), "2024-01-02".to_string()];
        let out = plain(&[s(&[f64::NAN, f64::NAN], &[])], &dates, &[]);

        assert!(out.svg.starts_with("<svg"));
        assert!(out.svg.contains("Jan"));
        assert_eq!(out.geom.n_local, 2);
    }

    #[test]
    fn history_lines_are_solid() {
        let out = plain(&[s(&[1.0, 2.0, 3.0], &[])], &[], &[]);
        // the only dasharray present should be absent (no forecast, no boundary)
        assert!(!out.svg.contains("stroke-dasharray=\"6 4\""));
    }

    #[test]
    fn renders_date_ticks_with_granularity() {
        let dates = vec!["2024-01-01".to_string(), "2024-06-01".to_string()];
        let out = line_chart_ex(
            &[s(&[1.0, 2.0], &[])],
            &dates,
            &[],
            None,
            false,
            &[],
            FeatureKindSet::none(),
            Granularity::Month,
        );
        assert!(out.svg.contains("Jan '24"));
    }

    #[test]
    fn geom_hit_test_maps_pixels_to_indices() {
        let out = plain(&[s(&[1.0, 2.0, 3.0, 4.0], &[])], &[], &[]);
        let g = out.geom;
        assert_eq!(g.index_at_vbx(PAD_L), 0);
        assert_eq!(g.index_at_vbx(W - PAD_R), 3);
    }

    #[test]
    fn view_range_zooms_and_sets_base_idx() {
        let out = line_chart_ex(
            &[s(&[1.0, 2.0, 3.0, 4.0, 5.0], &[])],
            &[],
            &[],
            Some((2, 4)),
            false,
            &[],
            FeatureKindSet::none(),
            Granularity::Day,
        );
        assert_eq!(out.geom.base_idx, 2);
        assert_eq!(out.geom.n_local, 3);
    }

    #[test]
    fn band_is_emitted_when_enabled() {
        let series = [ChartSeries {
            h: &[1.0, 2.0],
            f: &[3.0, 4.0],
            upper: Some(&[3.5, 4.5]),
            lower: Some(&[2.5, 3.5]),
            color_idx: 0,
        }];
        let out = line_chart_ex(
            &series,
            &[],
            &[],
            None,
            true,
            &[],
            FeatureKindSet::none(),
            Granularity::Day,
        );
        assert!(out.svg.contains("fill=\"var(--viz-uncertainty-fill)\""));
        assert_eq!(
            out.svg
                .matches("stroke=\"var(--viz-uncertainty-line)\"")
                .count(),
            2
        );
    }

    #[test]
    fn markers_drawn_only_for_enabled_kinds() {
        let marks = vec![vec![
            FeatureMarker {
                idx: 0,
                kind: FeatureKind::Anomaly,
                value: 1.0,
            },
            FeatureMarker {
                idx: 1,
                kind: FeatureKind::Holiday,
                value: 2.0,
            },
        ]];
        let mut en = FeatureKindSet::none();
        en = en.with_toggled(FeatureKind::Anomaly);
        let out = line_chart_ex(
            &[s(&[1.0, 2.0, 3.0], &[])],
            &[],
            &[],
            None,
            false,
            &marks,
            en,
            Granularity::Day,
        );
        assert!(out.svg.contains("<rect")); // anomaly square shown
        assert!(!out.svg.contains("<circle")); // holiday circle hidden
    }
}
