"""
Monsoon Analysis for AgriSentinal

Transparent, rainfall-based monsoon behavior analysis using historical data.
Estimates onset, variability, and early/normal/late classification.

All calculations are based on historical rainfall patterns only.
No IMD data used. No ML. No sensor data.
"""

import numpy as np
import pandas as pd


def calculate_monsoon_onset(rainfall_df, mm_threshold=20, min_consecutive=3):
    """
    Estimate monsoon onset from historical daily rainfall.

    Onset is defined as the first date where rainfall >= threshold mm
    sustained for at least min_consecutive days.

    Parameters
    ----------
    rainfall_df : pandas.DataFrame
        Must contain 'date' and 'rain_sum' columns (daily rainfall in mm).
        Must be sorted by date ascending.
    mm_threshold : float
        Minimum daily rainfall (mm) to count as monsoon start.
    min_consecutive : int
        Number of consecutive days above threshold to declare onset.

    Returns
    -------
    dict or None
        Onset information dict, or None if insufficient data.
    """
    if rainfall_df is None or rainfall_df.empty:
        return None

    df = rainfall_df.copy()
    if "date" not in df.columns or "rain_sum" not in df.columns:
        return None

    df = df.sort_values("date").reset_index(drop=True)

    # Find first day with rainfall >= threshold
    rainy_days = df[df["rain_sum"] >= mm_threshold]

    if rainy_days.empty:
        return {
            "onset_date": None,
            "onset_day_of_year": None,
            "classification": "Unknown",
            "note": "No rainfall >= {mm_threshold} mm found in historical data",
        }

    onset_day = rainy_days.iloc[0]["date"]
    doy = onset_day.timetuple().tm_yday if hasattr(onset_day, "timetuple") else None

    # Check for sustained rainfall (consecutive days)
    # Simple approach: count consecutive days from onset
    onset_idx = rainy_days.index[0]
    consecutive = 0
    for i in range(onset_idx, len(df)):
        if df.loc[i, "rain_sum"] >= mm_threshold:
            consecutive += 1
        else:
            break

    # Classify onset timing
    month = onset_day.month if hasattr(onset_day, "month") else None
    if month:
        if month <= 6:
            classification = "Early"
        elif month <= 7:
            classification = "Normal"
        else:
            classification = "Late"
    else:
        classification = "Unknown"

    return {
        "onset_date": onset_day.strftime("%Y-%m-%d") if hasattr(onset_day, "strftime") else None,
        "onset_day_of_year": doy,
        "classification": classification,
        "mm_threshold_used": mm_threshold,
        "consecutive_rainy_days": consecutive,
        "note": f"Onset defined as first day with >= {mm_threshold} mm rainfall sustained {consecutive} days",
    }


def calculate_typical_onset(rainfall_df, mm_threshold=20):
    """
    Calculate median/typical monsoon onset from multi-year historical data.

    Parameters
    ----------
    rainfall_df : pandas.DataFrame
        Historical daily rainfall. Must have 'date' and 'rain_sum' columns.
        Ideally contains multiple years of data.
    mm_threshold : float
        Rainfall threshold for onset definition.

    Returns
    -------
    dict or None
        Typical onset information, or None if insufficient data.
    """
    if rainfall_df is None or rainfall_df.empty:
        return None

    df = rainfall_df.copy()
    if "date" not in df.columns or "rain_sum" not in df.columns:
        return None

    # Group by month-day to find when 50% cumulative rainfall occurs each year
    # For simplicity with single-year data, find when cumulative rain reaches 50%
    df_sorted = df.sort_values("date").reset_index(drop=True)
    total_rain = df_sorted["rain_sum"].sum()

    if total_rain < mm_threshold:
        return {
            "typical_onset_doY": None,
            "classification": "Insufficient rainfall",
            "note": f"Total annual rainfall ({total_rain:.1f} mm) below threshold ({mm_threshold} mm)",
        }

    # Find day when 50% of total rainfall has occurred
    cumulative = 0
    onset_day_of_year = None
    for _, row in df_sorted.iterrows():
        cumulative += row["rain_sum"]
        if cumulative >= total_rain / 2:
            onset_day_of_year = row["date"].timetuple().tm_yday
            break

    if onset_day_of_year is None:
        return {
            "typical_onset_doY": None,
            "classification": "Could not determine",
            "note": "Cumulative rainfall analysis failed",
        }

    # Classify
    if onset_day_of_year <= 182:  # Jan 1 = day 1, Jun 30 ≈ day 182
        classification = "Early"
    elif onset_day_of_year <= 243:  # Jul 31 ≈ day 243
        classification = "Normal"
    else:
        classification = "Late"

    return {
        "typical_onset_day_of_year": onset_day_of_year,
        "typical_onset_date_approx": f"Day {onset_day_of_year}",
        "classification": classification,
        "total_annual_rainfall": float(total_rain),
        "note": "Median onset based on 50% of annual cumulative rainfall",
    }


def calculate_rainfall_anomaly(current_rainfall, historical_mean, historical_std=None):
    """
    Calculate rainfall anomaly as a percentage deviation from historical mean.

    Parameters
    ----------
    current_rainfall : float
        Rainfall amount for current period (mm).
    historical_mean : float
        Mean rainfall for same period from historical data (mm).
    historical_std : float, optional
        Standard deviation of historical rainfall (mm).

    Returns
    -------
    dict
        Anomaly analysis.
    """
    if historical_mean == 0:
        return {
            "anomaly_percentage": 0,
            "deviation_mm": current_rainfall,
            "relative_to": "historical mean of 0",
            "risk": "unknown",
        }

    anomaly_pct = ((current_rainfall - historical_mean) / historical_mean) * 100
    deviation_mm = current_rainfall - historical_mean

    # Assess risk level
    if historical_std is not None and historical_std > 0:
        z_score = (current_rainfall - historical_mean) / historical_std
        if z_score > 2:
            risk = "High excess"
        elif z_score > 1:
            risk = "Moderate excess"
        elif z_score < -2:
            risk = "High deficit"
        elif z_score < -1:
            risk = "Moderate deficit"
        else:
            risk = "Near normal"
    else:
        risk = "Unknown (limited variability data)"

    return {
        "anomaly_percentage": round(anomaly_pct, 1),
        "deviation_mm": round(deviation_mm, 1),
        "historical_mean_mm": round(historical_mean, 1),
        "current_rainfall_mm": round(current_rainfall, 1),
        "risk": risk,
        "note": "Positive anomaly = more rain than historical; Negative = less",
    }


def summarize_historical_rainfall(rainfall_df):
    """
    Summarize historical rainfall patterns from a DataFrame.

    Parameters
    ----------
    rainfall_df : pandas.DataFrame
        Daily rainfall with 'date' and 'rain_sum' columns.

    Returns
    -------
    dict
        Summary statistics.
    """
    if rainfall_df is None or rainfall_df.empty:
        return {
            "total_rainfall": 0,
            "rainy_days": 0,
            "max_daily_rain": 0,
            "mean_daily_rain": 0,
            "median_daily_rain": 0,
            "dry_spell_max": 0,
            "note": "No historical data available",
        }

    df = rainfall_df.copy()
    if "rain_sum" not in df.columns:
        return {
            "total_rainfall": 0,
            "rainy_days": 0,
            "max_daily_rain": 0,
            "mean_daily_rain": 0,
            "median_daily_rain": 0,
            "dry_spell_max": 0,
            "note": "No 'rain_sum' column in data",
        }

    total = df["rain_sum"].sum()
    rainy = (df["rain_sum"] > 0).sum()
    max_daily = df["rain_sum"].max()
    mean_daily = df["rain_sum"].mean()
    median_daily = df["rain_sum"].median()

    # Max consecutive dry days
    dry_streak = 0
    max_dry = 0
    for rain in df["rain_sum"]:
        if rain < 1.0:
            dry_streak += 1
            max_dry = max(max_dry, dry_streak)
        else:
            dry_streak = 0

    return {
        "total_rainfall": round(total, 1),
        "rainy_days": int(rainy),
        "max_daily_rain": round(max_daily, 1),
        "mean_daily_rain": round(mean_daily, 1),
        "median_daily_rain": round(median_daily, 1),
        "dry_spell_max_days": max_dry,
        "note": "Summary based on available historical daily rainfall",
    }


def classify_onsom_behavior(onset_day_of_year):
    """
    Classify monsoon onset behavior based on day-of-year.

    Parameters
    ----------
    onset_day_of_year : int or None
        Day of year (1-366) when onset occurred.

    Returns
    -------
    str
        "Early", "Normal", or "Late"
    """
    if onset_day_of_year is None:
        return "Unknown"

    if onset_day_of_year <= 182:  # End of June
        return "Early"
    elif onset_day_of_year <= 243:  # End of July
        return "Normal"
    else:
        return "Late"


# Example usage (remove or comment out in production)
if __name__ == "__main__":
    # Sample synthetic data for testing
    dates = pd.date_range("2023-06-01", periods=120, freq="D")
    # Simulate monsoon: very low rain in June, increasing in July
    rain = np.zeros(120, dtype=float)
    rain[30:60] = np.random.uniform(0, 5, 30)  # early spurious rain
    rain[60:90] = np.random.uniform(5, 20, 30)  # monsoon proper
    rain[90:120] = np.random.uniform(2, 10, 30)  # tail end

    df = pd.DataFrame({"date": dates, "rain_sum": rain})

    print("=== Monsoon Onset Detection ===")
    onset = calculate_monsoon_onset(df, mm_threshold=20)
    print(onset)

    print("\n=== Typical Onset (median) ===")
    typical = calculate_typical_onset(df, mm_threshold=20)
    print(typical)

    print("\n=== Rainfall Anomaly ===")
    anomaly = calculate_rainfall_anomaly(500.0, 400.0, 80.0)
    print(anomaly)

    print("\n=== Historical Summary ===")
    summary = summarize_historical_rainfall(df)
    print(summary)

    print("\n=== Onset Classification ===")
    cls = classify_onsom_behavior(onset["onset_day_of_year"]) if onset else "N/A"
    print(f"Onset classification: {cls}")