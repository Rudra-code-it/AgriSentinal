"""
Transparent Agronomic Decision Engine for AgriSentinal

Evaluates Early, Normal, and Late sowing windows using
agronomic rules (no ML). Returns scores + explanations
for each window and recommends the best.

Does NOT claim harvest/yield probability.
Scores represent estimated probability of favorable
crop-establishment conditions only.
"""

# Crop establishment thresholds (mm rainfall, % soil moisture, etc.)
CROP_ESTABLISHMENT = {
    "soybean": {"rain_mm": 50, "min_moisture_fraction": 0.25, "temp_opt": (25, 30)},
    "maize": {"rain_mm": 80, "min_moisture_fraction": 0.30, "temp_opt": (22, 28)},
    "rice": {"rain_mm": 100, "min_moisture_fraction": 0.60, "temp_opt": (20, 28)},
    "cotton": {"rain_mm": 50, "min_moisture_fraction": 0.20, "temp_opt": (21, 30)},
    "groundnut": {"rain_mm": 50, "min_moisture_fraction": 0.25, "temp_opt": (25, 30)},
    "pulses": {"rain_mm": 40, "min_moisture_fraction": 0.20, "temp_opt": (20, 28)},
}


def score_rainfall(window_rain_mm, crop_req_mm, current_anomaly):
    """Score rainfall adequacy for a sowing window.

    Returns (score_pct, reasons) where score_pct is 0-100.
    Positive anomalies near req boost score; large deficits reduce it.
    """
    if window_rain_mm >= crop_req_mm:
        base = 100
        reasons = ["Rainfall adequate for establishment"]
    else:
        shortfall_pct = (crop_req_mm - window_rain_mm) / crop_req_mm * 100
        base = max(0, 100 - shortfall_pct)
        reasons = ["Insufficient rainfall for establishment"]

    # Adjust for anomaly
    if abs(current_anomaly) <= 20:
        if current_anomaly > 0:
            base = min(100, base + 5)  # slight boost for above-normal
        else:
            base = min(100, base + 2)  # small bonus for near-normal
    elif current_anomaly > 40:
        base = max(0, base - 15)  # penalize excess rain
    elif current_anomaly < -40:
        base = max(0, base - 20)  # penalize deficit

    return round(base, 1), reasons


def score_soil_moisture(moisture_fraction, min_frac, optimal_frac):
    """Score soil moisture suitability for crop establishment.

    Returns (score_pct, reasons) where score_pct is 0-100.
    """
    if moisture_fraction >= optimal_frac:
        score = 100
        reasons = ["Soil moisture optimal for establishment"]
    elif moisture_fraction >= min_frac:
        # Linear interpolation: 50% at min, 100% at optimal
        score = 50 + ((moisture_fraction - min_frac) / (optimal_frac - min_frac)) * 50
        reasons = ["Soil moisture adequate for establishment"]
    elif moisture_fraction > 0:
        score = int(moisture_fraction / min_frac * 50)
        reasons = ["Soil moisture marginal - may require irrigation"]
    else:
        score = 0
        reasons = ["Insufficient soil moisture - irrigation required"]

    return round(score, 1), reasons


def score_dry_spell(dry_spell_days, crop_tolerance_days):
    """Score dry-spell risk for a sowing window.

    Returns (score_pct, reasons) where higher = lower risk.
    A score of 100 = no dry-spell risk.
    """
    if dry_spell_days <= crop_tolerance_days:
        score = 100 - (dry_spell_days / crop_tolerance_days) * 30
        reasons = ["Dry-spell risk within crop tolerance"]
    else:
        excess = dry_spell_days - crop_tolerance_days
        score = max(0, 70 - (excess * 5))
        reasons = ["Dry-spell exceeds crop tolerance"]

    return round(max(0, min(100, score)), 1), reasons


def score_temperature(temp_c, temp_opt_min, temp_opt_max):
    """Score temperature suitability for crop establishment.

    Returns (score_pct, reasons) where higher = more suitable.
    """
    if temp_c >= temp_opt_min and temp_c <= temp_opt_max:
        score = 100
        reasons = ["Temperature within optimal range"]
    elif temp_c < temp_opt_min:
        deficit = temp_opt_min - temp_c
        score = max(0, 100 - deficit)
        reasons = [f"Temperature below optimal (deficit {deficit}°C)"]
    else:  # temp_c > temp_opt_max
        excess = temp_c - temp_opt_max
        score = max(0, 100 - excess)
        reasons = [f"Temperature above optimal (excess {excess}°C)"]

    return round(score, 1), reasons


def score_crop_compatibility(monsoon_onset_class, window_type, crop):
    """Score how well the monsoon onset behavior matches the sowing window.

    Returns (score_pct, reasons).
    """
    # Crop-specific monsoon dependence
    dep = {"soybean": "high", "maize": "high", "rice": "very high",
           "cotton": "moderate to high", "groundnut": "moderate", "pulses": "moderate"}

    dependence = dep.get(crop, "moderate")

    if window_type == "Early":
        if monsoon_onset_class == "Early":
            score = 100
            reasons = ["Early monsoon aligns with early sowing"]
        elif monsoon_onset_class == "Normal":
            score = 70
            reasons = ["Monsoon timing marginally compatible"]
        else:  # Late
            score = 30
            reasons = ["Early sowing risks missing monsoon onset"]
    elif window_type == "Normal":
        if monsoon_onset_class == "Normal":
            score = 100
            reasons = ["Normal monsoon aligns with normal sowing"]
        elif monsoon_onset_class == "Early":
            score = 80
            reasons = ["Early monsoon benefits normal sowing"]
        else:  # Late
            score = 55
            reasons = ["Late monsoon may delay normal sowing"]
    else:  # Late window
        if monsoon_onset_class == "Late":
            score = 100
            reasons = ["Late monsoon aligns with late sowing"]
        elif monsoon_onset_class == "Early":
            score = 65
            reasons = ["Early monsoon may end before late sowing establishes"]
        else:  # Normal
            score = 75
            reasons = ["Normal monsoon generally compatible with late window"]

    return round(score, 1), reasons


def compute_window_scores(
    rainfall_mm,
    rainfall_anomaly,
    moisture_fraction,
    forecast_rain_mm,
    dry_spell_days,
    temp_c,
    monsoon_onset_class,
    crop="soybean",
):
    """Compute suitability scores for Early, Normal, and Late sowing windows.

    Returns a dict with scores for each window + explanations.
    """
    crop_req = CROP_ESTABLISHMENT.get(crop)
    if crop_req is None:
        crop_req = CROP_ESTABLISHMENT["soybean"]

    # Window definitions and their characteristics
    windows = {
        "Early": {
            "forecast_rain_adj": forecast_rain_mm * 0.7,  # early season often gets less rain
            "temp_opt_min": crop_req["temp_opt"][0] - 2,  # slightly more tolerant early
            "temp_opt_max": crop_req["temp_opt"][1] + 2,
            "dry_spell_tol": 4,  # crops often more tolerant early
        },
        "Normal": {
            "forecast_rain_adj": forecast_rain_mm,
            "temp_opt_min": crop_req["temp_opt"][0],
            "temp_opt_max": crop_req["temp_opt"][1],
            "dry_spell_tol": 3,
        },
        "Late": {
            "forecast_rain_adj": forecast_rain_mm * 1.1,  # late season can get excess
            "temp_opt_min": crop_req["temp_opt"][0] - 3,
            "temp_opt_max": crop_req["temp_opt"][1] + 3,
            "dry_spell_tol": 2,  # less tolerant late
        },
    }

    results = {}

    for window_name, cfg in windows.items():
        # 1. Rainfall score (using adjusted forecast rain for the window)
        rain_score, rain_reasons = score_rainfall(
            cfg["forecast_rain_adj"], crop_req["rain_mm"], rainfall_anomaly
        )

        # 2. Soil moisture score
        min_frac = crop_req.get("min_moisture_fraction", 0.25)
        opt_frac = 0.45  # default optimal fraction
        soil_score, soil_reasons = score_soil_moisture(moisture_fraction, min_frac, opt_frac)

        # 3. Dry-spell risk score
        ds_tol = cfg["dry_spell_tol"]
        ds_score, ds_reasons = score_dry_spell(dry_spell_days, ds_tol)

        # 4. Temperature score
        temp_min = cfg["temp_opt_min"]
        temp_max = cfg["temp_opt_max"]
        temp_score, temp_reasons = score_temperature(temp_c, temp_min, temp_max)

        # 5. Crop compatibility score (monsoon onset)
        comp_score, comp_reasons = score_crop_compatibility(monsoon_onset_class, window_name, crop)

        # 6. Establishment-condition suitability (weighted composite)
        # Weights: rainfall 40%, soil moisture 30%, temp 20%, dry-spell 10%
        establishment_score = round(
            0.4 * rain_score + 0.3 * soil_score + 0.2 * temp_score + 0.1 * ds_score
        )

        reasons = [
            f"Rainfall: {rain_reasons[0]}",
            f"Soil moisture: {soil_reasons[0]}",
            f"Temperature: {temp_reasons[0]}",
            f"Dry-spell: {ds_reasons[0]}",
            f"Compatibility: {comp_reasons[0]}",
        ]

        results[window_name] = {
            "suitability_score": establishment_score,
            "establishment_condition suitability": round(
                0.8 * establishment_score + 0.2 * comp_score
            ),
            "rainfall_score": rain_score,
            "rainfall adequacy": rain_reasons,
            "soil_moisture_score": soil_score,
            "soil-moisture suitability": soil_reasons,
            "dry-spell risk": ds_score,
            "dry-spell risk text": ds_reasons,
            "temperature risk": temp_score,
            "temperature risk text": temp_reasons,
            "crop compatibility": comp_score,
            "crop compatibility text": comp_reasons,
        }

    return results


def recommend_window(window_scores):
    """Recommend the highest-scoring window.

    Returns (recommended_window, explanation_dict).
    """
    if not window_scores:
        return None, {"reason": "No window data available"}

    # Find window with highest suitability score
    best_window = max(window_scores, key=lambda w: window_scores[w]["suitability_score"])
    best = window_scores[best_window]

    # Build explanation
    explanation = {
        "recommended_window": best_window,
        "suitability_probability": best["suitability_score"],  # NOT harvest probability
        "data_confidence": "Medium-High",
        "reasons_for_recommendation": [],
        "why_not_early": [],
        "why_not_late": [],
    }

    # Positive reasons for recommendation
    if best["rainfall_score"] >= 80:
        explanation["reasons_for_recommendation"].append(
            f"Rainfall adequate ({best['rainfall_score']}%)"
        )
    if best["soil_moisture_score"] >= 80:
        explanation["reasons_for_recommendation"].append(
            f"Soil moisture suitable ({best['soil_moisture_score']}%)"
        )
    if best["dry-spell risk"] >= 80:
        explanation["reasons_for_recommendation"].append(
            f"Low dry-spell risk ({best['dry-spell risk']}%)"
        )
    if best["temperature risk"] >= 80:
        explanation["reasons_for_recommendation"].append(
            f"Temperature suitable ({best['temperature risk']}%)"
        )

    # Explain why other windows are weaker
    for w in ["Early", "Late"]:
        if w != best_window and w in window_scores:
            other = window_scores[w]
            # Simple comparison based on key scores
            if other["rainfall_score"] < window_scores[best_window]["rainfall_score"] - 15:
                explanation["why_not_" + w.lower()].append(
                    f"{w} window has lower rainfall suitability"
                )
            if other["dry-spell risk"] < window_scores[best_window]["dry-spell risk"] - 15:
                explanation["why_not_" + w.lower()].append(
                    f"{w} window has higher dry-spell risk"
                )

    # If no specific reasons added, add generic
    if not explanation["reasons_for_recommendation"]:
        explanation["reasons_for_recommendation"].append(
            "Best overall establishment conditions across all windows"
        )

    # Only explain why other windows are weaker if they are not the recommended one
    if best_window != "Early" and not explanation["why_not_early"]:
        explanation["why_not_early"].append(
            "Early window scored lower on overall suitability"
        )
    if best_window != "Late" and not explanation["why_not_late"]:
        explanation["why_not_late"].append(
            "Late window scored lower on overall suitability"
        )

    return best_window, explanation


# Example usage (remove or comment out in production)
if __name__ == "__main__":
    # Example: simulate a field in Kharif season
    example_inputs = {
        "rainfall_mm": 65.0,           # recent/forecast rainfall
        "rainfall_anomaly": 15.0,      # % deviation from historical mean
        "moisture_fraction": 0.40,     # model-based estimated soil moisture
        "forecast_rain_mm": 40.0,      # near-term forecast
        "dry_spell_days": 3,           # consecutive dry days
        "temp_c": 28.0,                # current temperature
        "monsoon_onset_class": "Normal",  # Early/Normal/Late
        "crop": "soybean",
    }

    scores = compute_window_scores(**example_inputs)
    recommended, explanation = recommend_window(scores)

    print("=== Sowing Window Scores ===")
    for window, s in scores.items():
        print(f"{window}: suitability={s['suitability_score']}%, "
              f"establishment={s['establishment-condition suitability']}%")

    print(f"\nRecommended: {recommended}")
    print(f"Probability: {explanation['suitability_probability']}% "
          f"(NOT harvest/yield probability)")
    print(f"Reasons: {'; '.join(explanation['reasons_for_recommendation'])}")
    print(f"Why not early: {'; '.join(explanation['why_not_early'])}")
    print(f"Why not late: {'; '.join(explanation['why_not_late'])}")