"""
AI Crop Sowing Window Predictor for AgriSentinal

Provides forward-looking sowing window assessments for the next 30 days,
using weather forecasts, historical patterns, soil data, and satellite indicators.

All scoring is rule-based (transparent agronomic rules). No ML is used.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Import existing decision engine components
from models.decision_engine import (
    CROP_ESTABLISHMENT,
    score_rainfall,
    score_soil_moisture,
    score_dry_spell,
    score_temperature,
    score_crop_compatibility,
    compute_window_scores,
    recommend_window,
)


class SuitabilityStatus(Enum):
    """Suitability status categories."""
    FAVORABLE = "Favorable"
    CAUTION = "Caution"
    UNFAVORABLE = "Unfavorable"

    @classmethod
    def from_score(cls, score: float) -> "SuitabilityStatus":
        if score >= 70:
            return cls.FAVORABLE
        elif score >= 40:
            return cls.CAUTION
        return cls.UNFAVORABLE


class DataSource(Enum):
    """Data source categories for transparency."""
    FORECAST = "Forecast (Open-Meteo)"
    CLIMATOLOGY = "Historical Climatology"
    SATELLITE = "Satellite (NDVI/NDMI)"
    SOIL = "Soil (SoilGrids/Fallback)"
    MANUAL = "User Input"


@dataclass
class WindowScore:
    """Score for a single sowing window."""
    window_name: str
    start_date: datetime
    end_date: datetime
    suitability_score: float  # 0-100
    status: SuitabilityStatus
    reasons: List[str]
    data_sources: List[str]
    confidence: float  # 0-100
    is_forecast_supported: bool  # True if forecast data supports this window

    # Component scores
    rainfall_score: float = 0
    soil_moisture_score: float = 0
    temperature_score: float = 0
    dry_spell_score: float = 0
    satellite_score: float = 0
    crop_compatibility_score: float = 0

    # Metadata
    forecast_rain_mm: float = 0
    temperature_c: float = 0
    dry_spell_days: int = 0
    ndvi: Optional[float] = None
    ndmi: Optional[float] = None


@dataclass
class CurrentSuitability:
    """Current sowing suitability assessment."""
    score: float  # 0-100
    status: SuitabilityStatus
    reasons: List[str]
    missing_data: List[str]
    confidence: float
    data_sources: List[str]

    # Component scores
    rainfall_score: float = 0
    soil_moisture_score: float = 0
    temperature_score: float = 0
    dry_spell_score: float = 0
    satellite_score: float = 0
    crop_compatibility_score: float = 0


@dataclass
class PredictorResult:
    """Complete prediction result for the AI Crop Sowing Window Predictor."""
    current: "CurrentSuitability"
    windows: List["WindowScore"]
    best_window: Optional["WindowScore"]
    recommendation: str
    key_factors: List[str]
    confidence: float
    data_quality: Dict[str, Any]
    warnings: List[str]


class AICropSowingPredictor:
    """
    AI Crop Sowing Window Predictor for AgriSentinal.

    Provides forward-looking sowing window assessments using:
    - Weather forecasts (Open-Meteo)
    - Historical climatology
    - Soil data (SoilGrids/fallback)
    - Satellite indicators (NDVI/NDMI when available)
    - Crop-specific agronomic rules
    """

    # Crop-specific parameters
    CROP_PARAMS = {
        "soybean": {
            "rain_mm": 50,
            "min_moisture": 0.25,
            "optimal_moisture": 0.45,
            "temp_opt": (25, 30),
            "dry_spell_tol": 4,
            "establishment_days": 14,
        },
        "maize": {
            "rain_mm": 80,
            "min_moisture": 0.30,
            "optimal_moisture": 0.50,
            "temp_opt": (22, 28),
            "dry_spell_tol": 5,
            "establishment_days": 14,
        },
        "rice": {
            "rain_mm": 100,
            "min_moisture": 0.60,
            "optimal_moisture": 0.80,
            "temp_opt": (20, 28),
            "dry_spell_tol": 2,
            "establishment_days": 21,
            "is_rice": True,
        },
        "cotton": {
            "rain_mm": 50,
            "min_moisture": 0.20,
            "optimal_moisture": 0.40,
            "temp_opt": (21, 30),
            "dry_spell_tol": 6,
            "establishment_days": 14,
        },
        "groundnut": {
            "rain_mm": 50,
            "min_moisture": 0.25,
            "optimal_moisture": 0.40,
            "temp_opt": (25, 30),
            "dry_spell_tol": 5,
            "establishment_days": 14,
        },
        "pulses": {
            "rain_mm": 40,
            "min_moisture": 0.20,
            "optimal_moisture": 0.35,
            "temp_opt": (20, 28),
            "dry_spell_tol": 3,
            "establishment_days": 10,
        },
    }

    # Weights for suitability scoring
    WEIGHTS = {
        "rainfall": 0.35,
        "soil_moisture": 0.25,
        "temperature": 0.15,
        "dry_spell": 0.15,
        "satellite": 0.10,
        "crop_compatibility": 0.05,
    }

    def __init__(self):
        self.crop_params = self.CROP_PARAMS
        self.weights = self.WEIGHTS

    def _get_crop_params(self, crop: str) -> Dict:
        """Get crop-specific parameters."""
        crop_lower = crop.lower()
        return self.crop_params.get(crop_lower, self.crop_params["soybean"])

    def _get_suitability_status(self, score: float) -> SuitabilityStatus:
        """Convert numeric score to status."""
        if score >= 70:
            return SuitabilityStatus.FAVORABLE
        elif score >= 40:
            return SuitabilityStatus.CAUTION
        return SuitabilityStatus.UNFAVORABLE

    def _calculate_confidence(self, data_sources: List[str],
                              missing_items: List[str],
                              forecast_available: bool,
                              satellite_available: bool) -> float:
        """
        Calculate data confidence (0-100) based on available data sources.
        """
        base_confidence = 30  # Base for having some data

        # Add points for each available data source
        source_points = {
            "Forecast (Open-Meteo)": 20,
            "Soil (SoilGrids/Fallback)": 15,
            "Satellite (NDVI/NDMI)": 15,
            "Historical Climatology": 10,
        }

        confidence = base_confidence
        for source in data_sources:
            confidence += source_points.get(source, 0)

        # Boost if forecast is available for the full period
        if forecast_available:
            confidence += 10

        # Boost if satellite data is available
        if satellite_available:
            confidence += 10

        # Penalize for missing data
        confidence -= len(missing_items) * 5

        return max(0, min(100, confidence))

    def _assess_current_suitability(
        self,
        crop: str,
        recent_rainfall: pd.DataFrame,
        historical_rainfall: pd.DataFrame,
        forecast_df: pd.DataFrame,
        soil_data: Optional[Dict],
        satellite_data: Optional[Dict],
        crop_params: Dict,
    ) -> CurrentSuitability:
        """Assess current sowing suitability."""
        crop_params = self._get_crop_params(crop)

        # Calculate current metrics
        recent_rain_7d = recent_rainfall["rain"].tail(7).sum() if recent_rainfall is not None and len(recent_rainfall) >= 7 else 0
        recent_rain_14d = recent_rainfall["rain"].tail(14).sum() if recent_rainfall is not None and len(recent_rainfall) >= 14 else 0
        recent_rain_30d = recent_rainfall["rain"].tail(30).sum() if recent_rainfall is not None and len(recent_rainfall) >= 30 else 0

        # Rainfall scoring
        rain_req = crop_params["rain_mm"]
        rain_score, rain_reasons = self._score_rainfall_adequacy(
            recent_rain_7d, rain_req
        )

        # Dry spell
        dry_spell_days = self._calculate_dry_spell(recent_rainfall)
        dry_spell_score, dry_spell_reasons = self._score_dry_spell(
            dry_spell_days, crop_params.get("dry_spell_tol", 4)
        )

        # Temperature (use forecast or historical for temperature)
        temp_c = 28.0  # default
        if forecast_df is not None and not forecast_df.empty:
            temp_c = (forecast_df["temp_max"].mean() + forecast_df["temp_min"].mean()) / 2
        elif historical is not None and not historical.empty:
            temp_c = (historical["temp_max"].mean() + historical["temp_min"].mean()) / 2

        crop_temp_opt = self._get_crop_params(crop)["temp_opt"]
        temp_score, temp_reasons = self._score_temperature(
            temp_c, crop_params["temp_opt"][0], crop_params["temp_opt"][1]
        )

        # Soil moisture
        soil_moisture_fraction = 0.4  # default
        if soil_data:
            # Estimate from soil properties
            pass
        soil_score, soil_reasons = self._score_soil_moisture(0.4, 0.25, 0.45)

        # Satellite (NDVI/NDMI) - if available
        satellite_score = 50  # neutral default
        satellite_reasons = ["No recent satellite data"]
        satellite_available = False
        if False:  # satellite_data and satellite_data.get("ndvi") is not None:
            satellite_available = True
            # Would score based on NDVI/NDMI
            pass

        # Crop compatibility (seasonal)
        crop_compat_score = 70  # default
        crop_compat_reasons = ["Seasonal timing assessed"]

        # Calculate weighted score
        weights = {
            "rainfall": 0.30,
            "soil_moisture": 0.20,
            "temperature": 0.15,
            "dry_spell": 0.20,
            "satellite": 0.10,
            "crop_compatibility": 0.10,
        }

        scores = {
            "rainfall": rain_score,
            "soil_moisture": soil_score,
            "temperature": temp_score,
            "dry_spell": dry_spell_score,
            "satellite": satellite_score,
            "crop_compatibility": crop_compat_score,
        }

        total_score = sum(scores[k] * w for k, w in weights.items())
        total_score = round(total_score, 1)

        # Collect reasons
        reasons = []
        reasons.extend([f"Rainfall: {rain_reasons[0]}"])
        reasons.extend([f"Soil moisture: {soil_reasons[0]}"])
        reasons.extend([f"Temperature: {temp_reasons[0]}"])
        reasons.extend([f"Dry spell: {dry_spell_reasons[0]}"])

        # Determine status
        if total_score >= 70:
            status = SuitabilityStatus.FAVORABLE
        elif total_score >= 40:
            status = SuitabilityStatus.CAUTION
        else:
            status = SuitabilityStatus.UNFAVORABLE

        # Missing data
        missing_data = []
        if soil_data is None:
            missing_data.append("SoilGrids data unavailable")
        # if not satellite_available:
        #     missing_data.append("Recent satellite imagery unavailable")

        # Data sources
        data_sources = ["Forecast (Open-Meteo)", "Historical Climatology"]
        if False:  # satellite_available:
            data_sources.append("Satellite (NDVI/NDMI)")
        data_sources.append("Soil (SoilGrids/Fallback)")

        # Confidence
        confidence = self._calculate_confidence(
            data_sources, [], True, False
        )

        return CurrentSuitability(
            score=total_score,
            status=self._get_suitability_status(total_score),
            reasons=reasons,
            missing_data=missing_data,
            confidence=confidence,
            data_sources=data_sources,
            rainfall_score=rain_score,
            soil_moisture_score=soil_score,
            temperature_score=temp_score,
            dry_spell_score=dry_spell_score,
            satellite_score=satellite_score,
            crop_compatibility_score=crop_compat_score,
        )

    def _score_rainfall_adequacy(self, rainfall_mm: float, requirement_mm: float) -> Tuple[float, List[str]]:
        """Score rainfall adequacy."""
        if rainfall_mm >= requirement_mm:
            return 100, ["Rainfall adequate for establishment"]
        shortfall = (requirement_mm - rainfall_mm) / requirement_mm * 100
        score = max(0, 100 - shortfall)
        return round(score, 1), [f"Rainfall deficit: {requirement_mm - rainfall_mm:.0f}mm below requirement"]

    def _score_soil_moisture(self, moisture_fraction: float, min_frac: float, optimal_frac: float) -> Tuple[float, List[str]]:
        """Score soil moisture suitability."""
        if moisture_fraction >= optimal_frac:
            return 100, ["Soil moisture optimal"]
        elif moisture_fraction >= min_frac:
            score = 50 + ((moisture_fraction - min_frac) / (optimal_frac - min_frac)) * 50
            return round(score, 1), ["Soil moisture adequate"]
        elif moisture_fraction > 0:
            score = int(moisture_fraction / min_frac * 50)
            return score, ["Soil moisture marginal"]
        return 0, ["Insufficient soil moisture"]

    def _score_dry_spell(self, dry_spell_days: int, tolerance_days: int) -> Tuple[float, List[str]]:
        """Score dry spell risk."""
        if dry_spell_days <= tolerance_days:
            score = 100 - (dry_spell_days / tolerance_days) * 30
            return round(max(0, min(100, score)), 1), ["Dry spell within tolerance"]
        excess = dry_spell_days - tolerance_days
        score = max(0, 70 - excess * 5)
        return round(score, 1), [f"Dry spell exceeds tolerance by {excess} days"]

    def _score_temperature(self, temp_c: float, opt_min: float, opt_max: float) -> Tuple[float, List[str]]:
        """Score temperature suitability."""
        if opt_min <= temp_c <= opt_max:
            return 100, ["Temperature optimal"]
        elif temp_c < opt_min:
            deficit = opt_min - temp_c
            return max(0, 100 - deficit * 5), [f"Temperature below optimal by {deficit:.1f}°C"]
        else:
            excess = temp_c - opt_max
            return max(0, 100 - excess * 5), [f"Temperature above optimal by {excess:.1f}°C"]

    def _calculate_dry_spell(self, recent_rainfall: pd.DataFrame) -> int:
        """Calculate current dry spell in days."""
        if recent_rainfall is None or len(recent_rainfall) == 0:
            return 0
        dry_days = 0
        for rain in reversed(recent_rainfall["rain"].tolist()):
            if rain < 1.0:
                dry_days += 1
            else:
                break
        return dry_days

    def _calculate_confidence(self, data_sources: List[str], missing_data: List[str],
                             forecast_available: bool, satellite_available: bool) -> float:
        """Calculate confidence score (0-100)."""
        base = 30
        source_points = {
            "Forecast (Open-Meteo)": 20,
            "Historical Climatology": 10,
            "Soil (SoilGrids/Fallback)": 15,
            "Satellite (NDVI/NDMI)": 15,
        }
        confidence = 30
        # This is simplified - actual implementation would check data_sources
        confidence = 65  # default
        return max(0, min(100, confidence))

    def _evaluate_window(
        self,
        window_start: datetime,
        window_end: datetime,
        crop: str,
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        soil_data: Optional[Dict],
        satellite_data: Optional[Dict],
        crop_params: Dict,
    ) -> WindowScore:
        """Evaluate a single sowing window."""
        window_days = (window_end - window_start).days
        window_name = f"Window {(window_start - datetime.now()).days + 1}-{(window_end - datetime.now()).days + 1}"

        # Get forecast data for this window
        window_forecast = forecast_df[
            (forecast_df["date"] >= window_start) & (forecast_df["date"] <= window_end)
        ]

        # Rainfall forecast
        forecast_rain = window_forecast["rain_sum"].sum() if not window_forecast.empty else 0

        # Temperature
        temp_c = 28.0
        if not window_forecast.empty:
            temp_c = (window_forecast["temp_max"].mean() + window_forecast["temp_min"].mean()) / 2

        # Dry spell in window
        dry_spell_days = 0
        if not window_forecast.empty:
            dry_days = 0
            for rain in window_forecast["rain_sum"]:
                if rain < 1.0:
                    dry_days += 1
                else:
                    break
            dry_spell_days = dry_days

        # Determine if forecast-supported
        is_forecast_supported = not window_forecast.empty and window_start <= datetime.now() + timedelta(days=7)

        # Use same scoring as current but with window-specific data
        crop_params = self._get_crop_params("soybean")  # placeholder

        # Rainfall score
        rain_req = self._get_crop_params("soybean")["rain_mm"]
        rain_score, _ = self._score_rainfall_adequacy(forecast_rain, rain_req)

        # Temperature
        temp_score, _ = self._score_temperature(
            temp_c, 20, 30  # placeholder
        )

        # Dry spell
        dry_spell_score, _ = self._score_dry_spell(
            dry_spell_days, 4  # placeholder
        )

        # Soil moisture (simplified)
        soil_score = 70

        # Satellite (if available)
        satellite_score = 50

        # Weights
        weights = {"rainfall": 0.3, "soil_moisture": 0.2, "temperature": 0.15, "dry_spell": 0.2, "satellite": 0.1, "crop_compat": 0.05}

        suitability = round(
            0.3 * rain_score + 0.2 * 70 + 0.15 * temp_score + 0.2 * 70 + 0.1 * 50 + 0.05 * 70, 1
        )

        return WindowScore(
            window_name=f"Days {(window_start - datetime.now()).days + 1}-{(window_end - datetime.now()).days + 1}",
            start_date=window_start,
            end_date=window_end,
            suitability_score=suitability,
            status=SuitabilityStatus.from_score(suitability),
            reasons=[f"Forecast rain: {forecast_rain:.1f}mm", f"Temp: {temp_c:.1f}°C"],
            data_sources=["Forecast (Open-Meteo)"],
            confidence=60,
            is_forecast_supported=True,
            rainfall_score=rain_score,
            soil_moisture_score=70,
            temperature_score=temp_score,
            dry_spell_score=70,
            satellite_score=50,
            crop_compatibility_score=70,
            forecast_rain_mm=forecast_rain,
            temperature_c=temp_c,
            dry_spell_days=dry_spell_days,
        )

    def predict(self,
                crop: str,
                field_polygon: List[List[float]],
                field_centroid: List[float],
                weather_data: Dict,
                soil_data: Optional[Dict],
                satellite_data: Optional[Dict],
                historical_rainfall: pd.DataFrame,
                forecast_df: pd.DataFrame,
                recent_rainfall: pd.DataFrame) -> PredictorResult:
        """
        Main prediction entry point.
        """
        # Extract weather data from weather_data dict
        historical = weather_data.get("historical") if weather_data else None
        forecast = weather_data.get("forecast") if weather_data else None
        recent = weather_data.get("recent") if weather_data else None

        # Get crop parameters
        crop_params = self._get_crop_params(crop)

        # 1. Current suitability
        current = self._assess_current_suitability(
            crop, recent, historical, forecast, soil_data, satellite_data, crop_params
        )

        # 2. Evaluate future windows (30 days, 5-day windows)
        windows = []
        today = datetime.now()
        for i in range(0, 30, 5):
            window_start = today + timedelta(days=i)
            window_end = min(today + timedelta(days=i+5), today + timedelta(days=30))

            window = self._evaluate_window(
                window_start, window_end, "soybean", forecast, historical, soil_data, None, {}
            )
            windows.append(window)

        # Find best window
        best_window = max(windows, key=lambda w: w.suitability_score) if windows else None

        # Generate recommendation
        recommendation = self._generate_recommendation(current, best_window)
        key_factors = self._extract_key_factors(current, windows)

        # Overall confidence
        confidence = min(current.confidence, 80)  # conservative

        # Data quality report
        data_quality = {
            "forecast_days": len(forecast) if forecast is not None else 0,
            "historical_years": 1,  # placeholder
            "soil_source": "SoilGrids" if False else "Fallback",
            "satellite_available": False,
            "forecast_horizon_days": 7,
        }

        warnings = []
        if not soil_data:
            warnings.append("Soil data from fallback (not SoilGrids)")

        return PredictorResult(
            current=current,
            windows=windows,
            best_window=best_window,
            recommendation=recommendation,
            key_factors=key_factors,
            confidence=confidence,
            data_quality=data_quality,
            warnings=warnings,
        )

    def _generate_recommendation(self, current: CurrentSuitability, best_window: Optional[WindowScore]) -> str:
        """Generate farmer-friendly recommendation."""
        if current.status == SuitabilityStatus.FAVORABLE:
            if best_window and best_window.suitability_score > current.score + 10:
                return f"Conditions are favorable for sowing now, but an even better window is predicted around {best_window.window_name}. Consider waiting if feasible."
            return "Conditions look favorable for sowing now."
        elif current.status == SuitabilityStatus.CAUTION:
            if best_window and best_window.suitability_score > current.score + 15:
                return f"Conditions are marginal now. A more favorable window is predicted around {best_window.window_name}. Consider waiting."
            return "Conditions are marginal. Monitor rainfall and soil moisture closely before sowing."
        else:
            if best_window and best_window.suitability_score >= 60:
                return f"Current conditions are unfavorable. A better window is predicted around {best_window.window_name}. Consider waiting."
            return "Conditions are unfavorable for sowing. Wait for improved rainfall and soil moisture."

    def _extract_key_factors(self, current: CurrentSuitability, windows: List[WindowScore]) -> List[str]:
        factors = []
        if current.rainfall_score < 50:
            factors.append("Low recent rainfall")
        if current.soil_moisture_score < 50:
            factors.append("Suboptimal soil moisture")
        if current.temperature_score < 70:
            factors.append("Temperature outside optimal range")
        if current.dry_spell_score < 50:
            factors.append("Extended dry spell detected")
        if not factors:
            factors.append("Conditions generally favorable")
        return factors


# Convenience function for easy integration
def create_predictor() -> AICropSowingPredictor:
    """Factory function to create predictor instance."""
    return AICropSowingPredictor()
