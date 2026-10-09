"""
Soil Moisture Model for AgriSentinal
Calculates model-based estimated soil moisture from rainfall, ET, and runoff.
Does NOT claim sensor measurements.
"""

import numpy as np
import pandas as pd


def estimate_soil_moisture(
    previous_moisture_fraction,
    rainfall_mm,
    evapotranspiration_mm,
    soil_holding_capacity_mm,
    crop="soybean",
    days_since_last=1,
    runoff_factor=0.1,
):
    """Estimate model-based soil moisture fraction.
    
    Formula: previous_moisture + rainfall - ET - runoff
    Runoff accounts for excess water beyond soil capacity.
    
    All values are in mm unless specified.
    Returns soil moisture as a fraction of total holding capacity [0, 1].
    """
    # Use default 0.3 if previous moisture unavailable
    if previous_moisture_fraction is None:
        previous_moisture_fraction = 0.3
    # Crop-specific adjustments
    crop_adj = CROP_SOIL_REQUIREMENTS.get(crop, CROP_SOIL_REQUIREMENTS["soybean"])
    effective_holding = soil_holding_capacity_mm * crop_adj["depth"]

    # Calculate runoff: water exceeding available capacity becomes runoff
    available_capacity = effective_holding * (1 - previous_moisture_fraction)
    excess_rainfall = max(0, rainfall_mm - available_capacity)
    runoff = excess_rainfall * runoff_factor

    # New moisture balance
    new_moisture_mm = (
        previous_moisture_fraction * effective_holding
        + rainfall_mm
        - evapotranspiration_mm
        - runoff
    )

    # Clamp to realistic bounds [0, effective_holding]
    new_moisture_mm = max(0, min(effective_holding, new_moisture_mm))

    # Return as fraction
    new_moisture_fraction = new_moisture_mm / effective_holding

    return {
        "moisture_fraction": round(new_moisture_fraction, 3),
        "moisture_mm": round(new_moisture_mm, 1),
        "holding_capacity_mm": round(effective_holding, 1),
        "model_based": True,
        "note": "Model-based estimated soil moisture - calculated from rainfall - ET - runoff",
    }


def calculate_soil_moisture_trend(historical_rainfall, soil_holding_capacity=150):
    """Calculate soil moisture trend from historical rainfall.
    
    Analyzes recent rainfall patterns to determine if moisture is
    improving, stable, or declining.
    """
    if historical_rainfall is None or len(historical_rainfall) < 3:
        return {
            "trend": "stable",
            "change": 0,
            "direction": "unknown",
        }

    # Convert to array and calculate moving averages
    rain = np.array(historical_rainfall[-30:])  # Last 30 days
    if len(rain) < 3:
        return {"trend": "stable", "change": 0, "direction": "unknown"}

    # Simple trend: compare first half vs second half
    mid = len(rain) // 2
    if mid < 2:
        return {"trend": "stable", "change": 0, "direction": "unknown"}

    first_half_mean = np.mean(rain[:mid])
    second_half_mean = np.mean(rain[mid:])

    change = second_half_mean - first_half_mean

    # Determine trend direction
    if abs(change) < 1.0:
        trend = "stable"
        direction = "≈ no change"
    elif change > 0:
        trend = "improving"
        direction = f"+{change:.1f} mm rainfall increase"
    else:
        trend = "declining"
        direction = f"{abs(change):.1f} mm rainfall decrease"

    return {
        "trend": trend,
        "change": round(change, 1),
        "direction": direction,
    }


def assess_moisture_suitability(
    soil_moisture_fraction,
    crop="soybean",
    min_threshold=None,
    optimal_threshold=None,
):
    """Assess if soil moisture is suitable for crop establishment."""
    # Use default fractions if soil moisture unavailable
    if soil_moisture_fraction is None:
        return {
            "suitability": "Unknown",
            "percentage": 0,
            "moisture_fraction": None,
            "min_required": min_threshold or 0.25,
            "optimal": optimal_threshold or 0.45,
            "note": "Missing soil moisture data - using default thresholds",
        }
    
    requirements = CROP_SOIL_REQUIREMENTS.get(crop, CROP_SOIL_REQUIREMENTS["soybean"])

    if min_threshold is None:
        min_threshold = requirements["min_moisture"]
    if optimal_threshold is None:
        optimal_threshold = requirements["optimal_moisture"]

    # Determine suitability category
    if soil_moisture_fraction >= optimal_threshold:
        suitability = "Excellent"
        percentage = int((soil_moisture_fraction / optimal_threshold) * 100)
    elif soil_moisture_fraction >= min_threshold:
        suitability = "Good"
        # Linear interpolation between min and optimal
        percentage = int(
            50
            + ((soil_moisture_fraction - min_threshold)
               / (optimal_threshold - min_threshold))
            * 50
        )
    elif soil_moisture_fraction > 0:
        suitability = "Marginal"
        percentage = int((soil_moisture_fraction / min_threshold) * 50)
    else:
        suitability = "Insufficient"
        percentage = 0

    return {
        "suitability": suitability,
        "percentage": min(percentage, 100),
        "moisture_fraction": soil_moisture_fraction,
        "min_required": min_threshold,
        "optimal": optimal_threshold,
    }


# Crop-specific constants
CROP_SOIL_REQUIREMENTS = {
    "soybean": {"min_moisture": 0.25, "optimal_moisture": 0.45, "depth": 0.6},
    "maize": {"min_moisture": 0.30, "optimal_moisture": 0.50, "depth": 0.7},
    "rice": {"min_moisture": 0.60, "optimal_moisture": 0.85, "depth": 0.8},
    "cotton": {"min_moisture": 0.20, "optimal_moisture": 0.40, "depth": 0.5},
    "groundnut": {"min_moisture": 0.25, "optimal_moisture": 0.40, "depth": 0.5},
    "pulses": {"min_moisture": 0.20, "optimal_moisture": 0.35, "depth": 0.4},
}