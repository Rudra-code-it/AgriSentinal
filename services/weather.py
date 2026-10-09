"""
Open-Meteo Weather Service for AgriSentinal
Retrieves historical and forecast weather data for field-level analysis.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os


class WeatherService:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        # Daily variables supported by both forecast and archive endpoints
        self.daily_vars = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
            "sunshine_duration",
        ]
        # Hourly variables only for forecast
        self.hourly_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "precipitation_probability",
        ]

    def set_location(self, lat, lon):
        self.latitude = lat
        self.longitude = lon
        return self

    def _build_archive_params(self, start_date, end_date):
        """Build params for archive API."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": self.daily_vars,
            "timezone": "UTC",
        }

    def _build_forecast_params(self, days=7):
        """Build params for forecast API."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": self.daily_vars,
            "hourly": self.hourly_vars,
            "forecast_days": days,
            "timezone": "UTC",
        }

    def get_historical_weather(self, start_date=None, end_date=None):
        """Get historical weather data for the location.

        Note: Archive API only provides data up to yesterday.
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            # Archive API only has data up to yesterday
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        params = self._build_archive_params(start_date, end_date)
        try:
            response = requests.get(self.archive_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_historical(data)
        except requests.exceptions.RequestException as e:
            print(f"Weather API error: {e}")
            return None

    def get_forecast(self, days=7):
        """Get weather forecast for the location."""
        params = self._build_forecast_params(days)
        try:
            response = requests.get(f"{self.base_url}/forecast", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_forecast(data)
        except requests.exceptions.RequestException as e:
            print(f"Forecast API error: {e}")
            return None

    def get_recent_rainfall(self, days=30):
        """Get recent rainfall accumulation for specified days."""
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        params = self._build_archive_params(start_date, end_date)
        try:
            response = requests.get(self.archive_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return self._parse_recent_rainfall(data)
        except requests.exceptions.RequestException as e:
            print(f"Recent rainfall API error: {e}")
            return None

    def _parse_historical(self, data):
        """Parse historical weather data into DataFrame."""
        if not data or "daily" not in data:
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"],
            "rain_sum": data["daily"]["rain_sum"],
            "precipitation_sum": data["daily"]["precipitation_sum"],
            "rain_prob_max": data["daily"]["precipitation_probability_max"],
            "sunshine_duration": data["daily"].get("sunshine_duration", [None]*len(data["daily"]["time"])),
        })

        df["rain_anomaly"] = df["rain_sum"] - df["rain_sum"].rolling(30).mean()
        return df

    def _parse_forecast(self, data):
        """Parse forecast data into DataFrame."""
        if not data or "daily" not in data:
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "temp_max": data["daily"]["temperature_2m_max"],
            "temp_min": data["daily"]["temperature_2m_min"],
            "rain_sum": data["daily"]["rain_sum"],
            "precipitation_prob_max": data["daily"]["precipitation_probability_max"],
            "sunshine_duration": data["daily"].get("sunshine_duration", [None]*len(data["daily"]["time"])),
        })
        return df

    def _parse_recent_rainfall(self, data):
        """Parse recent rainfall data."""
        if not data or "daily" not in data:
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "rain": data["daily"]["rain_sum"],
        })
        # Calculate cumulative rainfall
        df["cumulative_rain"] = df["rain"].cumsum()
        return df


# Utility functions for weather-derived metrics
def calculate_dry_spell(rainfall_series, max_gap_days=3):
    """Calculate dry spell duration from rainfall series.

    A dry spell is defined as consecutive days with rainfall < 1mm.
    """
    if len(rainfall_series) < 2:
        return 0

    dry_days = 0
    max_dry = 0

    for rain in rainfall_series:
        if rain < 1.0:  # Less than 1mm rainfall
            dry_days += 1
            max_dry = max(max_dry, dry_days)
        else:
            dry_days = 0

    return max_dry


def calculate_rainfall_anomaly(current_rain, historical_avg):
    """Calculate rainfall anomaly as percentage."""
    if historical_avg == 0:
        return 0
    return ((current_rain - historical_avg) / historical_avg) * 100


def estimate_soil_moisture_change(
    previous_moisture,
    rainfall,
    evapotranspiration,
    runoff_factor=0.1,
    soil_holding_capacity=200,
):
    """Estimate model-based soil moisture change.

    Formula: previous_moisture + rainfall - ET - runoff
    Runoff is proportional to rainfall exceeding holding capacity.
    """
    # Runoff: water exceeding holding capacity becomes runoff
    excess_rainfall = max(0, rainfall - (soil_holding_capacity - previous_moisture))
    runoff = excess_rainfall * runoff_factor

    # New moisture balance
    new_moisture = previous_moisture + rainfall - evapotranspiration - runoff

    # Clamp to realistic bounds [0, holding capacity]
    new_moisture = max(0, min(soil_holding_capacity, new_moisture))

    return new_moisture


def assess_dry_spell_risk(
    forecast_rain,
    recent_dry_spell,
    crop_dry_spell_tolerance,
):
    """Assess dry-spell risk based on forecast and crop tolerance."""
    if forecast_rain is None:
        # No rain forecast - risk based on recent history
        if recent_dry_spell >= crop_dry_spell_tolerance:
            return "High"
        elif recent_dry_spell >= crop_dry_spell_tolerance * 0.5:
            return "Moderate"
        else:
            return "Low"

    # If rain is forecast, assess probability
    if forecast_rain > 10:  # Significant rain (>10mm)
        return "Low"
    elif forecast_rain > 0:
        return "Moderate"
    else:
        # No rain forecast, check recent history
        if recent_dry_spell >= crop_dry_spell_tolerance:
            return "High"
        else:
            return "Moderate"