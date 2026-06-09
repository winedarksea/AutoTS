//! Minimal dependency-free SVG line chart.
//!
//! Rendered as an inline SVG string. Every chart is paired with an accessible
//! DOM table elsewhere, so the SVG is decorative (aria-hidden) — screen readers
//! and LLMs read the structured table instead.

const W: f64 = 920.0;
const H: f64 = 340.0;
const PAD_L: f64 = 48.0;
const PAD_R: f64 = 16.0;
const PAD_T: f64 = 16.0;
const PAD_B: f64 = 38.0; // extra room for x-axis date labels

/// Skip hover groups on very dense charts (SVG size + usability).
const MAX_HOVER_PTS: usize = 300;

/// One line to draw: history portion (solid) + optional forecast portion (dashed),
/// both in the same palette color. `color_idx` maps to `--viz-s{n}` (wraps at 7).
pub struct ChartSeries<'a> {
    pub h: &'a [f64],
    pub f: &'a [f64],
    pub color_idx: usize,
}

impl ChartSeries<'_> {
    fn color(&self) -> String {
        format!("var(--viz-s{})", self.color_idx % 7)
    }
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

fn path_for(values: &[f64], x0: f64, n_total: usize, lo: f64, hi: f64) -> String {
    let plot_w = W - PAD_L - PAD_R;
    let plot_h = H - PAD_T - PAD_B;
    let xstep = if n_total > 1 {
        plot_w / (n_total as f64 - 1.0)
    } else {
        plot_w
    };
    let mut d = String::new();
    let mut pen_down = false;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            pen_down = false;
            continue;
        }
        let x = PAD_L + (x0 + i as f64) * xstep;
        let y = PAD_T + plot_h * (1.0 - (v - lo) / (hi - lo));
        if pen_down {
            d.push_str(&format!(" L{x:.1} {y:.1}"));
        } else {
            d.push_str(&format!(" M{x:.1} {y:.1}"));
            pen_down = true;
        }
    }
    d
}

/// Compact numeric label that fits in the tooltip box.
fn fmt_value(v: f64) -> String {
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

/// "2024-01-15 …" → "Jan '24". Falls back to the raw 10-char prefix on failure.
fn fmt_date_label(s: &str) -> String {
    let s = if s.len() > 10 { &s[..10] } else { s };
    let mut parts = s.splitn(3, '-');
    let year = parts.next().unwrap_or("");
    let month = parts.next().unwrap_or("");
    if year.len() < 4 || month.len() < 2 {
        return s.to_string();
    }
    let mon = match month {
        "01" => "Jan",
        "02" => "Feb",
        "03" => "Mar",
        "04" => "Apr",
        "05" => "May",
        "06" => "Jun",
        "07" => "Jul",
        "08" => "Aug",
        "09" => "Sep",
        "10" => "Oct",
        "11" => "Nov",
        "12" => "Dec",
        _ => return s.to_string(),
    };
    format!("{} '{}", mon, &year[2..])
}

/// Up to `k` evenly-spaced indices from `0..n`, deduped.
fn date_tick_indices(n: usize, k: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    if n <= k {
        return (0..n).collect();
    }
    let mut v: Vec<usize> = (0..k).map(|i| i * (n - 1) / (k - 1)).collect();
    v.dedup();
    v
}

/// SVG group for one interactive data point: invisible hit circle (CSS hover
/// triggers the inner `.chart-tt` tooltip). Rendered on top of paths so it
/// captures pointer events. `color` is already a CSS `var(…)` expression.
fn hover_group(px: f64, py: f64, value: f64, date: &str, color: &str) -> String {
    // keep tooltip box inside the SVG horizontally
    let tx = px.clamp(PAD_L + 38.0, W - PAD_R - 38.0);
    // tooltip above point; flip below if near the top
    let box_top = if py > PAD_T + 52.0 { py - 46.0 } else { py + 8.0 };
    let val_str = fmt_value(value);
    let date_str = if date.is_empty() {
        String::new()
    } else {
        fmt_date_label(date)
    };
    format!(
        "<g class=\"chart-pt\">\
         <circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"7\" fill=\"none\" pointer-events=\"all\"/>\
         <g class=\"chart-tt\" pointer-events=\"none\">\
           <circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"3.5\" fill=\"{color}\"/>\
           <rect class=\"chart-tt-bg\" x=\"{:.1}\" y=\"{box_top:.1}\" width=\"76\" height=\"38\" rx=\"5\"/>\
           <text class=\"chart-tt-val\" x=\"{tx:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{val_str}</text>\
           <text class=\"chart-tt-date\" x=\"{tx:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{date_str}</text>\
         </g>\
        </g>",
        tx - 38.0,
        box_top + 15.0,
        box_top + 29.0,
    )
}

/// Build an SVG for one or more series sharing a common date axis.
///
/// Each `ChartSeries` carries its own history and forecast value slices plus a
/// palette index; all share `history_dates` / `forecast_dates` for the x-axis.
/// The y-axis is auto-scaled to the combined range of every visible series.
///
/// - History portions are solid lines; forecast portions are dashed with
///   mosaic-diamond markers — both use the same series color.
/// - Up to 5 minimal "Mon 'YY" date labels appear on the x-axis.
/// - Each data point gets an invisible hit area; hovering shows a value+date
///   tooltip (CSS `.chart-pt:hover .chart-tt`). Skipped when n > MAX_HOVER_PTS.
pub fn line_chart(
    series: &[ChartSeries<'_>],
    history_dates: &[String],
    forecast_dates: &[String],
) -> String {
    if series.is_empty() {
        return String::new();
    }

    let n_history = series.iter().map(|s| s.h.len()).max().unwrap_or(0);
    let n_forecast = series.iter().map(|s| s.f.len()).max().unwrap_or(0);
    let n_total = n_history + n_forecast;
    if n_total == 0 {
        return String::new();
    }

    // Y scale: combine all series values.
    let all_slices: Vec<&[f64]> = series.iter().flat_map(|s| [s.h, s.f]).collect();
    let (lo, hi) = match finite_min_max(&all_slices) {
        Some(v) => v,
        None => return String::new(),
    };

    let all_dates: Vec<&str> = history_dates
        .iter()
        .chain(forecast_dates.iter())
        .map(|s| s.as_str())
        .collect();

    let plot_w = W - PAD_L - PAD_R;
    let plot_h = H - PAD_T - PAD_B;
    let xstep = if n_total > 1 {
        plot_w / (n_total as f64 - 1.0)
    } else {
        plot_w
    };

    let mut svg = format!(
        "<svg viewBox=\"0 0 {W} {H}\" role=\"img\" aria-hidden=\"true\" \
         xmlns=\"http://www.w3.org/2000/svg\">"
    );

    // y gridlines + labels (5 ticks)
    for t in 0..=4 {
        let frac = t as f64 / 4.0;
        let y = PAD_T + plot_h * frac;
        let val = hi - (hi - lo) * frac;
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

    // x-axis date labels — up to 5 minimal ticks
    let axis_y = H - PAD_B;
    if !all_dates.is_empty() {
        for &idx in &date_tick_indices(n_total, 5) {
            if let Some(&date_str) = all_dates.get(idx) {
                if date_str.is_empty() {
                    continue;
                }
                let x = PAD_L + idx as f64 * xstep;
                let lx = x.clamp(PAD_L + 26.0, W - PAD_R - 26.0);
                let label = fmt_date_label(date_str);
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
    }

    // history/forecast boundary marker (once, shared across all series)
    if n_history > 0 && n_forecast > 0 {
        let xb = PAD_L + (n_history as f64 - 0.5) * xstep;
        svg.push_str(&format!(
            "<line x1=\"{xb:.1}\" y1=\"{PAD_T:.0}\" x2=\"{xb:.1}\" y2=\"{axis_y:.1}\" \
             stroke=\"var(--md-outline)\" stroke-dasharray=\"4 4\" stroke-width=\"1\"/>",
        ));
    }

    // Paths for every series (history solid, forecast dashed — same color each)
    for s in series {
        let color = s.color();
        if !s.h.is_empty() {
            let d = path_for(s.h, 0.0, n_total, lo, hi);
            svg.push_str(&format!(
                "<path d=\"{d}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>"
            ));
        }
        if !s.f.is_empty() {
            let (forecast_path_values, x0) = if let Some(&last_h) = s.h.last() {
                let mut values = Vec::with_capacity(s.f.len() + 1);
                values.push(last_h);
                values.extend_from_slice(s.f);
                (values, s.h.len() as f64 - 1.0)
            } else {
                (s.f.to_vec(), 0.0)
            };
            let d = path_for(&forecast_path_values, x0, n_total, lo, hi);
            svg.push_str(&format!(
                "<path d=\"{d}\" fill=\"none\" stroke=\"{color}\" \
                 stroke-width=\"2.5\" stroke-dasharray=\"6 4\"/>"
            ));
            // Mosaic-tile diamond markers on each forecast point
            let fx0 = if s.h.is_empty() { 0.0 } else { s.h.len() as f64 };
            for (i, &v) in s.f.iter().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                let x = PAD_L + (fx0 + i as f64) * xstep;
                let y = PAD_T + plot_h * (1.0 - (v - lo) / (hi - lo));
                let r = 3.5;
                svg.push_str(&format!(
                    "<path d=\"M{x:.1} {:.1} L{:.1} {y:.1} L{x:.1} {:.1} L{:.1} {y:.1} Z\" \
                     fill=\"{color}\"/>",
                    y - r,
                    x + r,
                    y + r,
                    x - r
                ));
            }
        }
    }

    // Interactive hover groups rendered last (on top of all paths + markers)
    if n_total <= MAX_HOVER_PTS {
        for s in series {
            let color = s.color();
            for (i, &v) in s.h.iter().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                let x = PAD_L + i as f64 * xstep;
                let y = PAD_T + plot_h * (1.0 - (v - lo) / (hi - lo));
                let date = all_dates.get(i).copied().unwrap_or("");
                svg.push_str(&hover_group(x, y, v, date, &color));
            }
            let fx0 = if s.h.is_empty() { 0.0 } else { s.h.len() as f64 };
            for (i, &v) in s.f.iter().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                let x = PAD_L + (fx0 + i as f64) * xstep;
                let y = PAD_T + plot_h * (1.0 - (v - lo) / (hi - lo));
                let date = all_dates.get(n_history + i).copied().unwrap_or("");
                svg.push_str(&hover_group(x, y, v, date, &color));
            }
        }
    }

    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::{line_chart, ChartSeries};

    #[test]
    fn chart_distinguishes_history_and_forecast() {
        let chart = line_chart(
            &[ChartSeries {
                h: &[1.0, 2.0],
                f: &[3.0, 4.0],
                color_idx: 0,
            }],
            &[],
            &[],
        );
        // dashed forecast line present; boundary dash present
        assert!(chart.contains("stroke-dasharray=\"6 4\""));
        assert!(chart.contains("stroke-dasharray=\"4 4\""));
    }

    #[test]
    fn multi_series_renders_both_colors() {
        let chart = line_chart(
            &[
                ChartSeries { h: &[1.0, 2.0], f: &[], color_idx: 0 },
                ChartSeries { h: &[3.0, 4.0], f: &[], color_idx: 1 },
            ],
            &[],
            &[],
        );
        assert!(chart.contains("var(--viz-s0)"));
        assert!(chart.contains("var(--viz-s1)"));
    }

    #[test]
    fn chart_renders_date_labels() {
        let dates = vec!["2024-01-01".to_string(), "2024-06-01".to_string()];
        let chart = line_chart(
            &[ChartSeries { h: &[1.0, 2.0], f: &[], color_idx: 0 }],
            &dates,
            &[],
        );
        assert!(chart.contains("Jan '24"));
    }

    #[test]
    fn chart_renders_hover_groups() {
        let chart = line_chart(
            &[ChartSeries { h: &[1.0, 2.0], f: &[], color_idx: 0 }],
            &[],
            &[],
        );
        assert!(chart.contains("chart-pt"));
        assert!(chart.contains("chart-tt"));
    }
}
