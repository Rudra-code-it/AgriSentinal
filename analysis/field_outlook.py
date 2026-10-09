import numpy as np

CROP_ESTABLISHMENT_RISK = {
    "soybean": {"min_rain_mm": 50, "max_dry_spell": 4, "optimal_temp": (25, 30)},
    "maize": {"min_rain_mm": 80, "max_dry_spell": 5, "optimal_temp": (22, 28)},
    "rice": {"min_rain_mm": 100, "max_dry_spell": 2, "optimal_temp": (20, 28)},
    "cotton": {"min_rain_mm": 50, "max_dry_spell": 7, "optimal_temp": (21, 30)},
    "groundnut": {"min_rain_mm": 50, "max_dry_spell": 5, "optimal_temp": (25, 30)},
    "pulses": {"min_rain_mm": 40, "max_dry_spell": 3, "optimal_temp": (20, 28)},
}


def assess_rainfall_trend(recent_rainfall_7d, recent_rainfall_14d, recent_rainfall_30d):
    change_7to14 = recent_rainfall_14d - recent_rainfall_7d if recent_rainfall_7d else 0
    change_14to30 = recent_rainfall_30d - recent_rainfall_14d if recent_rainfall_14d else 0
    avg_recent = (recent_rainfall_7d + recent_rainfall_14d) / 2 if (recent_rainfall_7d and recent_rainfall_14d) else 0
    
    if avg_recent >= 60:
        return "Favorable", ["Above-average rainfall in recent period"]
    elif avg_recent >= 30:
        return "Normal", ["Average rainfall in recent period"]
    else:
        return "Deficient", ["Below-average rainfall in recent period"]


def assess_dry_spell_risk(dry_spell_days, crop_name):
    crop_tol = CROP_ESTABLISHMENT_RISK.get(crop_name, CROP_ESTABLISHMENT_RISK["soybean"])
    max_dry = crop_tol["max_dry_spell"]
    
    if dry_spell_days <= max_dry * 0.5:
        return "Low", ["Dry-spell activity within crop tolerance"]
    elif dry_spell_days <= max_dry:
        return "Moderate", ["Dry-spell activity at acceptable levels"]
    else:
        excess = dry_spell_days - max_dry
        return "High", [f"Prolonged dry-spell exceeds crop tolerance by {excess} day(s)"]


def assess_temperature_risk(temp_c, crop_name):
    crop_tol = CROP_ESTABLISHMENT_RISK.get(crop_name, CROP_ESTABLISHMENT_RISK["soybean"])
    opt_min, opt_max = crop_tol["optimal_temp"]
    
    if temp_c >= opt_min and temp_c <= opt_max:
        return "Low", ["Temperature within optimal range for establishment"]
    elif temp_c < opt_min:
        deficit = opt_min - temp_c
        if deficit <= 5:
            return "Moderate", [f"Temperature below optimal by {deficit}°C - slow establishment"]
        else:
            return "Moderate", [f"Temperature significantly below optimal by {deficit}°C"]
    else:
        excess = temp_c - opt_max
        if excess <= 3:
            return "Moderate", [f"Temperature above optimal by {excess}°C - mild heat stress"]
        else:
            return "High", [f"Temperature well above optimal by {excess}°C - significant heat stress"]


def assess_soil_moisture_trend(moisture_fraction_prev, moisture_fraction_curr):
    change = moisture_fraction_curr - moisture_fraction_prev
    
    if change > 0.05:
        return "Improving", ["Soil moisture increasing - favorable for establishment"]
    elif change < -0.05:
        return "Declining", ["Soil moisture decreasing - may require attention"]
    else:
        return "Stable", ["Soil moisture relatively stable"]


def assess_crop_establishment_risk(rainfall_mm, dry_spell_days, temp_c, moisture_fraction, crop_name):
    crop_tol = CROP_ESTABLISHMENT_RISK.get(crop_name, CROP_ESTABLISHMENT_RISK["soybean"])
    min_rain = crop_tol["min_rain_mm"]
    max_dry = crop_tol["max_dry_spell"]
    
    rain_ok = rainfall_mm >= min_rain
    dry_ok = dry_spell_days <= max_dry
    temp_min, temp_max = crop_tol["optimal_temp"]
    temp_ok = temp_min <= temp_c <= temp_max
    moisture_ok = moisture_fraction >= 0.25
    
    unfavorable = sum([not rain_ok, not dry_ok, not temp_ok, not moisture_ok])
    
    if unfavorable == 0:
        return "Favorable", ["All establishment conditions met"]
    elif unfavorable <= 1:
        return "Moderate Risk", ["Some establishment conditions sub-optimal"]
    else:
        return "High Risk", ["Multiple establishment conditions concerning"]


def calculate_data_confidence(forecast_available, historical_data_quality, 
                               fallback_used, data_agreement=None):
    confidence_score = 0
    
    if forecast_available:
        confidence_score += 30
    
    if historical_data_quality == "good":
        confidence_score += 30
    elif historical_data_quality == "fair":
        confidence_score += 15
    
    if fallback_used:
        confidence_score -= 20
    
    if data_agreement is not None and data_agreement:
        confidence_score += 10
    
    confidence_score = max(0, min(100, confidence_score))
    
    if confidence_score >= 80:
        return "High"
    elif confidence_score >= 50:
        return "Medium"
    else:
        return "Low"


def generate_field_outlook(
    forecast_7d=None,
    forecast_14d=None,
    forecast_30d=None,
    historical_rainfall_7d=None,
    historical_rainfall_14d=None,
    historical_rainfall_30d=None,
    dry_spell_days=None,
    temp_c=None,
    moisture_fraction_prev=None,
    moisture_fraction_curr=None,
    crop_name="soybean",
    forecast_covers_full_period=False,
    fallback_used=False,
):
    if historical_rainfall_7d is None:
        historical_rainfall_7d = 15.0
    if historical_rainfall_14d is None:
        historical_rainfall_14d = 30.0
    if historical_rainfall_30d is None:
        historical_rainfall_30d = 55.0
    if dry_spell_days is None:
        dry_spell_days = 3
    if temp_c is None:
        temp_c = 28.0
    if crop_name not in CROP_ESTABLISHMENT_RISK:
        crop_name = "soybean"
    
    rainfall_trend, rainfall_reasons = assess_rainfall_trend(
        historical_rainfall_7d, historical_rainfall_14d, historical_rainfall_30d
    )
    
    dry_spell_risk, dry_spell_reasons = assess_dry_spell_risk(dry_spell_days, crop_name)
    
    temp_risk, temp_reasons = assess_temperature_risk(temp_c, crop_name)
    
    moisture_trend, moisture_reasons = assess_soil_moisture_trend(moisture_fraction_prev, moisture_fraction_curr)
    
    total_rain = (historical_rainfall_7d or 0) + (historical_rainfall_14d or 0) + (historical_rainfall_30d or 0)
    establishment_risk, establishment_reasons = assess_crop_establishment_risk(
        total_rain, dry_spell_days, temp_c, moisture_fraction_curr, crop_name
    )
    
    all_reasons = []
    all_reasons.extend(rainfall_reasons[:2])
    all_reasons.extend(dry_spell_reasons[:1])
    all_reasons.extend(temp_reasons[:1])
    all_reasons.extend(moisture_reasons[:1])
    all_reasons.extend(establishment_reasons[:1])
    reasons = all_reasons[:5]
    
    forecast_available = forecast_covers_full_period or (forecast_7d is not None)
    historical_quality = "good" if all(x is not None for x in [historical_rainfall_7d, historical_rainfall_14d, historical_rainfall_30d]) else "fair"
    data_confidence = calculate_data_confidence(
        forecast_available=forecast_available,
        historical_data_quality=historical_quality,
        fallback_used=fallback_used,
        data_agreement=True
    )
    
    outlook_text = (
        f"**20-25 Day Field Risk Outlook**\n\n"
        f"**Rainfall Trend:** {rainfall_trend}\n"
        f"- {'; '.join(rainfall_reasons)}\n\n"
        f"**Dry-Spell Risk:** {dry_spell_risk}\n"
        f"- {'; '.join(dry_spell_reasons)}\n\n"
        f"**Temperature/Heat-Stress Risk:** {temp_risk}\n"
        f"- {'; '.join(temp_reasons)}\n\n"
        f"**Model-Based Soil-Moisture Trend:** {moisture_trend}\n"
        f"- {'; '.join(moisture_reasons)}\n\n"
        f"**Crop Establishment Risk:** {establishment_risk}\n"
        f"- {'; '.join(establishment_reasons)}\n\n"
        f"***\n"
    )
    
    disclaimer = "This is an actionable risk outlook based on forecast, historical patterns and model-based estimates. It is not an exact 20–25 day weather forecast."
    
    if not forecast_covers_full_period:
        outlook_text += f"*Outlook based on available near-term forecast ({forecast_7d or 0}d) supplemented by historical patterns and model-based trends for the remaining period.*"
    
    if fallback_used:
        outlook_text += f"*Some data uses fallback/cached values - see data confidence assessment below.*"
    
    result = {
        "20-25 Day Field Risk Outlook": outlook_text,
        "Data Confidence": data_confidence
    }
    
    return result
