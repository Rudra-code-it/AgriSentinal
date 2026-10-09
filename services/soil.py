"""
Soil Service for AgriSentinal
Retrieves soil information from SoilGrids with fallback to local data.
"""

import json
import os
from pathlib import Path

import numpy as np


class SoilInfo:
    """Hold soil information for a field."""

    def __init__(self, soil_type="loam", texture="medium",
                 water_holding_capacity=150, drainage_class="moderate",
                 organic_matter=1.5, bulk_density=1.4):
        self.soil_type = soil_type
        self.texture = texture
        self.water_holding_capacity = water_holding_capacity  # mm of plant-available water per meter depth
        self.drainage_class = drainage_class
        self.organic_matter = organic_matter  # % by weight
        self.bulk_density = bulk_density  # g/cm3


class SoilService:
    """Service for retrieving soil information."""

    def __init__(self):
        self.soil_grid_available = self._check_soilgrids_availability()
        self.fallback_soils = self._load_fallback_soils()

    def _check_soilgrids_availability(self):
        """Check if SoilGrids API is available."""
        # SoilGrids requires registration; for this prototype we use fallback
        # In production, would check for API credentials
        has_creds = bool(
            os.environ.get("SOILGRIDS_USER")
            and os.environ.get("SOILGRIDS_PASS")
        )
        return has_creds  # Set to False for prototype with fallback

    def _load_fallback_soils(self):
        """Load local fallback soil dataset for India."""
        return {
            "alluvial": SoilInfo(
                soil_type="Alluvial",
                texture="Loam",
                water_holding_capacity=180,
                drainage_class="Moderate to good",
                organic_matter=0.8,
                bulk_density=1.5,
            ),
            "red": SoilInfo(
                soil_type="Red",
                texture="Sandy loam",
                water_holding_capacity=120,
                drainage_class="Moderate",
                organic_matter=1.2,
                bulk_density=1.45,
            ),
            "black": SoilInfo(
                soil_type="Black Cotton",
                texture="Clay",
                water_holding_capacity=200,
                drainage_class="Poor to moderate",
                organic_matter=1.5,
                bulk_density=1.35,
            ),
            "laterite": SoilInfo(
                soil_type="Laterite",
                texture="Sandy clay",
                water_holding_capacity=100,
                drainage_class="Moderate to poor",
                organic_matter=0.6,
                bulk_density=1.55,
            ),
            "mountain": SoilInfo(
                soil_type="Mountain/Forest",
                texture="Loamy sand",
                water_holding_capacity=80,
                drainage_class="Good",
                organic_matter=2.0,
                bulk_density=1.3,
            ),
        }

    def identify_soil_type(self, latitude, longitude):
        """Identify soil type for a given location using SoilGrids or fallback."""
        if self.soil_grid_available:
            return self._get_soilgrids_soil(latitude, longitude)
        else:
            # Use fallback: simple regional mapping based on location
            # In a full implementation, this would use geo-coordinate mapping
            return self._fallback_soil_by_region(latitude, longitude)

    def _fallback_soil_by_region(self, lat, lon):
        """Determine soil type by region for India."""
        # Very rough regional mapping for demonstration
        if 20 <= lat <= 30 and 70 <= lon <= 80:
            # North India - mix of alluvial and red
            soil_key = "alluvial"
        elif 8 <= lat <= 25 and 72 <= lon <= 85:
            # East India - alluvial
            soil_key = "alluvial"
        elif 6 <= lat <= 22 and 68 <= lon <= 78:
            # Central India - mix
            soil_key = "black"
        else:
            # Default
            soil_key = "alluvial"

        return self.fallback_soils.get(soil_key, self.fallback_soils["alluvial"])

    def get_soil_info(self, soil_identifier=None, latitude=None, longitude=None):
        """Get soil information by identifier or location."""
        if soil_identifier:
            soil = self.fallback_soils.get(
                soil_identifier.lower(), self.fallback_soils["alluvial"]
            )
        elif latitude is not None and longitude is not None:
            soil = self.identify_soil_type(latitude, longitude)
        else:
            soil = self.fallback_soils["alluvial"]

        return {
            "soil_type": soil.soil_type,
            "texture": soil.texture,
            "water_holding_capacity": soil.water_holding_capacity,
            "drainage_class": soil.drainage_class,
            "organic_matter": soil.organic_matter,
            "bulk_density": soil.bulk_density,
            "source": "fallback_local" if not self.soil_grid_available else "soilgrids_api",
        }


# Crop-specific soil moisture requirements
CROP_SOIL_REQUIREMENTS = {
    "soybean": {"min_moisture": 0.25, "optimal_moisture": 0.45, "depth": 0.6},  # fraction of holding capacity
    "maize": {"min_moisture": 0.30, "optimal_moisture": 0.50, "depth": 0.7},
    "rice": {"min_moisture": 0.60, "optimal_moisture": 0.85, "depth": 0.8},  # Rice needs more water
    "cotton": {"min_moisture": 0.20, "optimal_moisture": 0.40, "depth": 0.5},
    "groundnut": {"min_moisture": 0.25, "optimal_moisture": 0.40, "depth": 0.5},
    "pulses": {"min_moisture": 0.20, "optimal_moisture": 0.35, "depth": 0.4},
}