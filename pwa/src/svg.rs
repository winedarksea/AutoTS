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
const PAD_B: f64 = 28.0;

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

/// Build an SVG for an optional history segment followed by an optional
/// forecast segment (sharing one x-axis). `forecast` is drawn in the primary
/// color with point markers; `history` in a muted gray.
pub fn line_chart(history: Option<&[f64]>, forecast: Option<&[f64]>) -> String {
    let h = history.unwrap_or(&[]);
    let f = forecast.unwrap_or(&[]);
    let n_total = h.len() + f.len();
    if n_total == 0 {
        return String::new();
    }

    let (lo, hi) = match finite_min_max(&[h, f]) {
        Some(v) => v,
        None => return String::new(),
    };

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
            "<text x=\"4\" y=\"{:.1}\" font-size=\"11\" fill=\"var(--md-on-surface-variant)\">{:.2}</text>",
            y + 4.0,
            val
        ));
    }

    // forecast/history boundary marker
    if !h.is_empty() && !f.is_empty() {
        let xb = PAD_L + (h.len() as f64 - 0.5) * xstep;
        svg.push_str(&format!(
            "<line x1=\"{xb:.1}\" y1=\"{PAD_T:.0}\" x2=\"{xb:.1}\" y2=\"{:.0}\" \
             stroke=\"var(--md-outline)\" stroke-dasharray=\"4 4\" stroke-width=\"1\"/>",
            H - PAD_B
        ));
    }

    if !h.is_empty() {
        let d = path_for(h, 0.0, n_total, lo, hi);
        svg.push_str(&format!(
            "<path d=\"{d}\" fill=\"none\" stroke=\"var(--md-outline)\" stroke-width=\"1.5\"/>"
        ));
    }
    if !f.is_empty() {
        // Include the final actual in the forecast path so the transition is
        // connected, while forecast markers remain on future-period positions.
        let (forecast_path_values, x0) = if let Some(last_actual) = h.last() {
            let mut values = Vec::with_capacity(f.len() + 1);
            values.push(*last_actual);
            values.extend_from_slice(f);
            (values, h.len() as f64 - 1.0)
        } else {
            (f.to_vec(), 0.0)
        };
        let d = path_for(&forecast_path_values, x0, n_total, lo, hi);
        svg.push_str(&format!(
            "<path d=\"{d}\" fill=\"none\" stroke=\"var(--md-primary)\" stroke-width=\"2.5\"/>"
        ));
        // markers
        for (i, &v) in f.iter().enumerate() {
            if !v.is_finite() {
                continue;
            }
            let forecast_x0 = if h.is_empty() { 0.0 } else { h.len() as f64 };
            let x = PAD_L + (forecast_x0 + i as f64) * xstep;
            let y = PAD_T + plot_h * (1.0 - (v - lo) / (hi - lo));
            svg.push_str(&format!(
                "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"3\" fill=\"var(--md-primary)\"/>"
            ));
        }
    }

    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::line_chart;

    #[test]
    fn chart_distinguishes_history_and_forecast() {
        let chart = line_chart(Some(&[1.0, 2.0]), Some(&[3.0, 4.0]));
        assert!(chart.contains("stroke=\"var(--md-outline)\""));
        assert!(chart.contains("stroke=\"var(--md-primary)\""));
        assert!(chart.contains("stroke-dasharray=\"4 4\""));
    }
}
