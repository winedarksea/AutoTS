//! Frequency inference and frequency-aware date-label formatting.
//!
//! The datetime axis arrives as ISO-ish strings ("YYYY-MM-DD" or
//! "YYYY-MM-DD HH:MM:SS"). AutoTS knows the true frequency, but it is not
//! threaded back to the frontend, so we infer a display granularity from the
//! median gap between consecutive points and format labels accordingly:
//! yearly shows only the year, monthly the month+year, daily the weekday+date,
//! hourly/minutely down to the clock time.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Granularity {
    Year,
    Month,
    Week,
    Day,
    Hour,
    Minute,
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
// 0 = Sunday (1970-01-01 was a Thursday; see weekday()).
const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

struct Parts {
    y: i64,
    mo: u32,
    d: u32,
    h: u32,
    mi: u32,
    se: u32,
}

/// Parse "YYYY-MM-DD[ T]HH:MM:SS" (time optional). Lenient: missing pieces → 0.
fn parse(s: &str) -> Option<Parts> {
    let s = s.trim();
    let (date, time) = match s.find(['T', ' ']) {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    let mut dp = date.splitn(3, '-');
    let y: i64 = dp.next()?.parse().ok()?;
    let mo: u32 = dp.next().unwrap_or("1").parse().unwrap_or(1);
    let d: u32 = dp.next().unwrap_or("1").parse().unwrap_or(1);
    let mut tp = time.split(':');
    let h: u32 = tp.next().unwrap_or("0").parse().unwrap_or(0);
    let mi: u32 = tp.next().unwrap_or("0").parse().unwrap_or(0);
    // seconds may carry a fractional part ("00.000") — keep the integer head.
    let se: u32 = tp
        .next()
        .and_then(|s| s.split('.').next())
        .unwrap_or("0")
        .parse()
        .unwrap_or(0);
    Some(Parts { y, mo: mo.clamp(1, 12), d: d.clamp(1, 31), h, mi, se })
}

/// Days since 1970-01-01 (Howard Hinnant's days_from_civil). Valid for any date.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = m as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn to_seconds(p: &Parts) -> i64 {
    days_from_civil(p.y, p.mo, p.d) * 86400 + (p.h * 3600 + p.mi * 60 + p.se) as i64
}

fn weekday(p: &Parts) -> &'static str {
    let days = days_from_civil(p.y, p.mo, p.d);
    // 1970-01-01 is Thursday; index 0 = Sunday.
    let idx = (((days % 7) + 4) % 7 + 7) % 7;
    WEEKDAYS[idx as usize]
}

/// Infer display granularity from the median gap between consecutive points.
pub fn infer_granularity(dates: &[String]) -> Granularity {
    let secs: Vec<i64> = dates.iter().filter_map(|s| parse(s).map(|p| to_seconds(&p))).collect();
    if secs.len() < 2 {
        return Granularity::Day;
    }
    let mut deltas: Vec<i64> = secs.windows(2).map(|w| (w[1] - w[0]).abs()).filter(|&d| d > 0).collect();
    if deltas.is_empty() {
        return Granularity::Day;
    }
    deltas.sort_unstable();
    let med = deltas[deltas.len() / 2];
    const DAY: i64 = 86400;
    if med >= 360 * DAY {
        Granularity::Year
    } else if med >= 27 * DAY {
        Granularity::Month
    } else if med >= 7 * DAY {
        Granularity::Week
    } else if med >= DAY {
        Granularity::Day
    } else if med >= 3600 {
        Granularity::Hour
    } else {
        Granularity::Minute
    }
}

/// Format one datetime string at the given granularity. Falls back to the raw
/// 10-char prefix if the string can't be parsed.
pub fn fmt_date(s: &str, g: Granularity) -> String {
    let Some(p) = parse(s) else {
        return s.chars().take(10).collect();
    };
    let mon = MONTHS[(p.mo - 1) as usize];
    match g {
        Granularity::Year => format!("{}", p.y),
        Granularity::Month => format!("{} {}", mon, p.y),
        Granularity::Week | Granularity::Day => {
            format!("{} {} {} {}", weekday(&p), p.d, mon, p.y)
        }
        Granularity::Hour | Granularity::Minute => {
            format!("{} {} {:02}:{:02}", p.d, mon, p.h, p.mi)
        }
    }
}

/// Compact axis-tick label (narrower than the tooltip's `fmt_date`).
pub fn fmt_tick(s: &str, g: Granularity) -> String {
    let Some(p) = parse(s) else {
        return s.chars().take(10).collect();
    };
    let mon = MONTHS[(p.mo - 1) as usize];
    match g {
        Granularity::Year => format!("{}", p.y),
        Granularity::Month => format!("{} '{:02}", mon, (p.y.rem_euclid(100))),
        Granularity::Week | Granularity::Day => format!("{} {}", p.d, mon),
        Granularity::Hour | Granularity::Minute => format!("{:02}:{:02}", p.h, p.mi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_monthly() {
        let d = vec![
            "2024-01-01".to_string(),
            "2024-02-01".to_string(),
            "2024-03-01".to_string(),
        ];
        assert_eq!(infer_granularity(&d), Granularity::Month);
    }

    #[test]
    fn infers_hourly() {
        let d = vec![
            "2024-01-01 00:00:00".to_string(),
            "2024-01-01 01:00:00".to_string(),
            "2024-01-01 02:00:00".to_string(),
        ];
        assert_eq!(infer_granularity(&d), Granularity::Hour);
    }

    #[test]
    fn weekday_is_correct() {
        // 2024-01-15 was a Monday.
        let p = parse("2024-01-15").unwrap();
        assert_eq!(weekday(&p), "Mon");
    }

    #[test]
    fn formats_per_granularity() {
        assert_eq!(fmt_date("2024-01-15", Granularity::Year), "2024");
        assert_eq!(fmt_date("2024-01-15", Granularity::Month), "Jan 2024");
        assert_eq!(fmt_date("2024-01-15", Granularity::Day), "Mon 15 Jan 2024");
        assert_eq!(fmt_date("2024-01-15 14:30:00", Granularity::Hour), "15 Jan 14:30");
    }
}
