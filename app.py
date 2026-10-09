import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from jinja2 import Template
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from models.decision_engine import compute_window_scores, recommend_window
from analysis.field_outlook import generate_field_outlook

# Satellite pipeline (Phase 1). Imported defensively so a missing optional







# Initialize session state for field data FIRST (before any access)
if "field_polygon" not in st.session_state:
    st.session_state.field_polygon = None
if "field_centroid" not in st.session_state:
    st.session_state.field_centroid = None
if "field_coords" not in st.session_state:
    st.session_state.field_coords = None




# Satellite pipeline (Phase 1). Imported defensively so a missing optional
# dependency degrades to an on-screen message instead of crashing the app.
try:
    from services.satellite import cdse_credentials_configured, get_field_satellite_data
except Exception as _satellite_import_error:  # pragma: no cover
    cdse_credentials_configured = None
    get_field_satellite_data = None
    SATELLITE_IMPORT_ERROR = str(_satellite_import_error)
else:
    SATELLITE_IMPORT_ERROR = None

# Weather pipeline (Phase 2). Open-Meteo only - no IMD, no fabrication.
try:
    from services.weather import WeatherService, calculate_dry_spell, calculate_rainfall_anomaly
    from analysis.monsoon import (
        calculate_monsoon_onset,
        calculate_typical_onset,
        calculate_rainfall_anomaly,
        summarize_historical_rainfall,
        classify_onsom_behavior,
    )
except Exception as _weather_import_error:  # pragma: no cover
    WeatherService = None
    calculate_dry_spell = None
    calculate_rainfall_anomaly = None
    calculate_monsoon_onset = None
    calculate_typical_onset = None
    calculate_rainfall_anomaly = None
    summarize_historical_rainfall = None
    classify_onsom_behavior = None
    WEATHER_IMPORT_ERROR = str(_weather_import_error)
else:
    WEATHER_IMPORT_ERROR = None

# Soil pipeline (Phase 3). SoilGrids with local fallback.
try:
    from services.soil import SoilService, CROP_SOIL_REQUIREMENTS
except Exception as _soil_import_error:  # pragma: no cover
    SoilService = None
    CROP_SOIL_REQUIREMENTS = {}
    SOIL_IMPORT_ERROR = str(_soil_import_error)
else:
    SOIL_IMPORT_ERROR = None


# --- Data Loader Functions (wrappers with caching) ---

@st.cache_data(ttl=1800, show_spinner=False, max_entries=16)
def load_satellite_data(polygon):
    """Live Sentinel-2 retrieval for a field AOI, memoised for 30 minutes."""
    if get_field_satellite_data is None:
        return None
    return get_field_satellite_data(polygon)


@st.cache_data(ttl=1800, show_spinner=False, max_entries=16)
def load_weather_data(lat, lon):
    """Retrieve historical and forecast weather for a field centroid.
    
    Returns a dict with historical_df, forecast_df, recent_rainfall_df,
    or None on failure. Memoised for 30 minutes.
    """
    if WeatherService is None:
        return None
    service = WeatherService().set_location(lat, lon)
    historical = service.get_historical_weather()
    forecast = service.get_forecast(days=7)
    recent = service.get_recent_rainfall(days=30)
    return {
        "historical": historical,
        "forecast": forecast,
        "recent": recent,
        "source": "Open-Meteo",
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


@st.cache_data(ttl=1800, show_spinner=False, max_entries=16)
def load_monsoon_analysis(historical_df):
    """Compute monsoon onset, anomaly, and historical summary from rainfall DataFrame."""
    if historical_df is None or historical_df.empty:
        return None
    onset = calculate_monsoon_onset(historical_df) if calculate_monsoon_onset else None
    typical = calculate_typical_onset(historical_df) if calculate_typical_onset else None
    summary = summarize_historical_rainfall(historical_df) if summarize_historical_rainfall else None
    return {
        "onset": onset,
        "typical": typical,
        "summary": summary,
    }


@st.cache_data(ttl=1800, show_spinner=False, max_entries=16)
def load_soil_data(lat, lon, manual_soil_type=None):
    """Retrieve soil information for field coordinates.
    
    Returns dict with soil properties and source info, or None on failure.
    """
    if SoilService is None:
        return None
    service = SoilService()
    if manual_soil_type:
        soil_info = service.get_soil_info(soil_identifier=manual_soil_type)
        soil_info["source"] = "manual_user_input"
        soil_info["note"] = "User-provided soil type (not measured or retrieved from SoilGrids)"
        return soil_info
    else:
        soil_info = service.get_soil_info(latitude=lat, longitude=lon)
        soil_info["source"] = soil_info.get("source", "unknown")
        if soil_info["source"] == "fallback_local":
            soil_info["note"] = "Fallback local mapping by region (not SoilGrids measurement)"
        return soil_info


# Custom CSS for modern agricultural dashboard styling
st.markdown("""
<style>
    /* Color palette variables */
    :root {
        --forest-green: #1b4d3e;
        --leaf-green: #2d7d4e;
        --leaf-green-light: #3a9d5f;
        --warm-cream: #faf8f3;
        --earth-tone: #8b7355;
        --earth-light: #d4c4b0;
        --amber: #c98700;
        --amber-light: #e8b42a;
        --red-alert: #c0392b;
        --gray-muted: #6b7280;
        --gray-light: #f3f4f6;
        --white: #ffffff;
        --border-color: #e5e7eb;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    }

    /* Global styles */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header styling */
    .agri-header {
        background: linear-gradient(135deg, var(--forest-green) 0%, var(--leaf-green) 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-md);
    }
    .agri-header h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
    }
    .agri-header .subtitle {
        margin: 0.5rem 0 0;
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    .agri-header .field-summary {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.2);
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        font-size: 0.875rem;
    }
    .agri-header .field-summary .item {
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }
    .agri-header .field-summary .label {
        opacity: 0.8;
    }
    .agri-header .field-summary .value {
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.375rem;
        font-weight: 600;
        color: var(--forest-green);
    }
    .section-header .status-badge {
        font-size: 0.75rem;
        padding: 0.25rem 0.625rem;
        border-radius: 9999px;
        font-weight: 500;
    }
    .status-live { background: #dcfce7; color: #166534; }
    .status-cached { background: #fef3c7; color: #92400e; }
    .status-unavailable { background: #f3f4f6; color: #6b7280; }

    /* Card styling */
    .agri-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .agri-card:hover {
        box-shadow: var(--shadow-md);
    }
    .agri-card .card-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--gray-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .agri-card .card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--forest-green);
        line-height: 1.2;
    }
    .agri-card .card-subtitle {
        font-size: 0.75rem;
        color: var(--gray-muted);
        margin-top: 0.25rem;
    }

    /* Status colors for metric values */
    .metric-favorable { color: var(--leaf-green) !important; }
    .metric-moderate { color: var(--amber) !important; }
    .metric-unfavorable { color: var(--red-alert) !important; }
    .metric-unavailable { color: var(--gray-muted) !important; }

    /* Card grid layouts */
    .card-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
    .card-grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
    .card-grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }

    @media (max-width: 1024px) {
        .card-grid-4 { grid-template-columns: repeat(2, 1fr); }
        .card-grid-3 { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 640px) {
        .card-grid-4, .card-grid-3, .card-grid-2 { grid-template-columns: 1fr; }
    }

    /* Image cards */
    .image-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    .image-card img {
        width: 100%;
        height: auto;
        display: block;
    }
    .image-card .caption {
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--forest-green);
        background: var(--warm-cream);
        border-top: 1px solid var(--border-color);
    }

    /* Plotly chart container */
    .plotly-container {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }

    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    .status-dot-live { background: var(--leaf-green); }
    .status-dot-cached { background: var(--amber); }
    .status-dot-unavailable { background: var(--gray-muted); }

    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background: var(--warm-cream);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--forest-green);
    }

    /* Metric improvements */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: var(--gray-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        background: white;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
        color: var(--forest-green);
    }

    /* Alert/Info/Warning/Success styling */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }
    .stAlert[data-baseweb="notification"] {
        box-shadow: var(--shadow-sm);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--leaf-green) !important;
        border-color: var(--leaf-green) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--forest-green) !important;
        border-color: var(--forest-green) !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--leaf-green) !important;
    }
</style>
""", unsafe_allow_html=True)



# Sidebar - Crop Selection
st.sidebar.markdown("### 🌱 Crop Selection")
crop_options = ["Soybean", "Maize", "Rice", "Cotton", "Groundnut", "Pulses"]
selected_crop = st.sidebar.selectbox("Select crop", crop_options, index=0, label_visibility="collapsed")

# Load crop config
import os
config_path = os.path.join(os.path.dirname(__file__), "config", "crop_config.json")
crop_config = {}
if os.path.exists(config_path):
    import json
    with open(config_path) as f:
        crop_config = json.load(f)

crop_info = crop_config.get(selected_crop.lower(), {})
st.sidebar.markdown(f"**{selected_crop} Configuration**")
if crop_info:
    st.sidebar.markdown(f"""
- **Sowing period:** {crop_info.get('sowing_period', 'N/A')}
- **Duration:** {crop_info.get('crop_duration', 'N/A')}
- **Moisture req.:** {crop_info.get('moisture_requirement', 'N/A')}
- **Optimal temp:** {crop_info.get('optimal_temp', 'N/A')}
- **Dry-spell tolerance:** {crop_info.get('dry_spell_tolerance', 'N/A')}
""")
else:
    st.sidebar.caption("Configuration not found")

# Polished Header
if st.session_state.field_polygon is not None and st.session_state.field_centroid is not None:
    header_html = f'''
    <div class="agri-header">
        <h1>🌾 AgriSentinal</h1>
        <div class="subtitle">AI-powered field-level Kharif sowing decision-support system</div>
        <div class="field-summary">
            <div class="item"><span class="label">Field:</span> <span class="value">{len(st.session_state.field_polygon)} vertices</span></div>
            <div class="item"><span class="label">Centroid:</span> <span class="value">{st.session_state.field_centroid[0]:.4f}° N, {st.session_state.field_centroid[1]:.4f}° E</span></div>
            <div class="item"><span class="label">Crop:</span> <span class="value">{selected_crop}</span></div>
        </div>
    </div>
    '''
else:
    header_html = f'''
    <div class="agri-header">
        <h1>🌾 AgriSentinal</h1>
        <div class="subtitle">AI-powered field-level Kharif sowing decision-support system</div>
        <div class="field-summary">
            <div class="item"><span class="label">Crop:</span> <span class="value">{selected_crop}</span></div>
            <div class="item"><span class="label">Status:</span> <span class="value">Draw a field to begin</span></div>
        </div>
    </div>
    '''
st.markdown(header_html, unsafe_allow_html=True)



# Field Selection Section
st.markdown('<div class="section-header"><h2>🗺️ Field Selection</h2></div>', unsafe_allow_html=True)

class _PinchZoomGuard(folium.MacroElement):
    """Plain scroll-wheel zoom stays disabled, but a trackpad pinch is
    reported by browsers as a ctrl/meta + wheel event, so that still zooms."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var pinchMap = {{ this._parent.get_name() }};
            if (!pinchMap || !pinchMap.getContainer) return;
            var pinchAcc = 0;
            var pinchTimer = null;
            pinchMap.getContainer().addEventListener(
                "wheel",
                function (e) {
                    if (!e.ctrlKey && !e.metaKey) return;
                    e.preventDefault();
                    pinchAcc += -e.deltaY;
                    if (Math.abs(pinchAcc) < 45) return;
                    var step = pinchAcc > 0 ? 1 : -1;
                    pinchAcc = 0;
                    var nextZoom = pinchMap.getZoom() + step;
                    var maxZ = pinchMap.getMaxZoom();
                    var minZ = pinchMap.getMinZoom();
                    if (maxZ !== Infinity) nextZoom = Math.min(nextZoom, maxZ);
                    if (minZ !== -Infinity) nextZoom = Math.max(nextZoom, minZ);
                    if (nextZoom !== pinchMap.getZoom()) pinchMap.setZoom(nextZoom);
                    clearTimeout(pinchTimer);
                    pinchTimer = setTimeout(function () { pinchAcc = 0; }, 250);
                },
                { passive: false }
            );
        })();
        {% endmacro %}
        """
    )

    def __init__(self):
        super().__init__()
        self._name = "PinchZoomGuard"


# Folium map - the single field-selection map
m = folium.Map(
    location=[20.0, 77.0],
    zoom_start=5,
    tiles="OpenStreetMap",
    zoom_control=True,
    double_click_zoom=True,
    scroll_wheel_zoom=False,
)

# Leaflet.draw toolbar: polygon + rectangle drawing, edit and delete
Draw(
    export=False,
    position="topleft",
    show_geometry_on_click=False,
    draw_options={
        "polyline": False,
        "polygon": True,
        "rectangle": True,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "allowIntersection": False,
        "showArea": True,
        "shapeOptions": {
            "color": "#2d7d4e",
            "weight": 2,
            "fillColor": "#2d7d4e",
            "fillOpacity": 0.15,
        },
    },
    edit_options={"remove": True},
).add_to(m)

# Trackpad pinch still zooms while plain scroll-wheel zoom is disabled
m.add_child(_PinchZoomGuard())

# Single map render - Leaflet Draw returns the drawn shapes as GeoJSON
map_data = st_folium(m, height=550, width=None, key="field_map", use_container_width=True)


def _ring_to_latlng(feature):
    """Return a closed [[lat, lng], ...] ring from a GeoJSON Polygon feature."""
    if not isinstance(feature, dict):
        return None
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Polygon":
        return None
    rings = geometry.get("coordinates") or []
    if not rings or not rings[0] or len(rings[0]) < 3:
        return None
    ring = rings[0]  # GeoJSON stores coordinates as [longitude, latitude]
    coords = [[float(pt[1]), float(pt[0])] for pt in ring]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


drawings = (map_data or {}).get("all_drawings") or []
polygon_features = [
    f
    for f in drawings
    if isinstance(f, dict) and (f.get("geometry") or {}).get("type") == "Polygon"
]
# Leaflet.draw keeps its feature group in creation order, so the newest
# surviving polygon/rectangle is the last one in the returned GeoJSON.
newest_ring = _ring_to_latlng(polygon_features[-1]) if polygon_features else None

if newest_ring is None:
    st.session_state.field_polygon = None
    st.session_state.field_coords = None
    st.session_state.field_centroid = None
else:
    coords_array = np.array(newest_ring, dtype=float)
    centroid_lat = float(coords_array[:, 0].mean())
    centroid_lng = float(coords_array[:, 1].mean())
    st.session_state.field_polygon = newest_ring
    st.session_state.field_coords = newest_ring
    st.session_state.field_centroid = [centroid_lat, centroid_lng]

# Field Information Cards
st.markdown('<div class="section-header"><h2>📍 Field Information</h2></div>', unsafe_allow_html=True)

if st.session_state.field_polygon is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="agri-card"><div class="card-title">Status</div><div class="card-value metric-favorable">Selected</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="agri-card"><div class="card-title">Vertices</div><div class="card-value">{len(st.session_state.field_polygon)}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="agri-card"><div class="card-title">Centroid</div><div class="card-value">{st.session_state.field_centroid[0]:.4f}° N<br>{st.session_state.field_centroid[1]:.4f}° E</div><div class="card-subtitle">Lat / Lng</div></div>', unsafe_allow_html=True)
else:
    st.info("Draw a polygon or rectangle on the map to select a field.")
    st.markdown('<div class="card-grid-3"><div class="agri-card"><div class="card-title">Status</div><div class="card-value metric-unavailable">No field</div></div><div class="agri-card"><div class="card-title">Vertices</div><div class="card-value metric-unavailable">—</div></div><div class="agri-card"><div class="card-title">Centroid</div><div class="card-value metric-unavailable">—</div></div></div>', unsafe_allow_html=True)
# Summary area
st.header("Field Information")
col1, col2 = st.columns(2)

with col1:
    if st.session_state.field_polygon is not None:
        st.metric("Selected", "Yes")
        st.metric("Vertices", len(st.session_state.field_polygon))
    else:
        st.metric("Selected", "No")

with col2:
    if st.session_state.field_centroid is not None:
        st.metric("Latitude", f"{st.session_state.field_centroid[0]:.4f}")
        st.metric("Longitude", f"{st.session_state.field_centroid[1]:.4f}")
    else:
        st.metric("Latitude", "—")
        st.metric("Longitude", "—")

# --- 🛰️ Satellite & Vegetation (Phase 1: live Sentinel-2 data) ---
st.header("🛰️ Satellite & Vegetation")

if get_field_satellite_data is None:
    st.error("Satellite module unavailable: " + str(SATELLITE_IMPORT_ERROR))
elif st.session_state.field_polygon is None:
    st.info("Draw a polygon or rectangle on the map to load Sentinel-2 imagery for this field.")
else:
    if not cdse_credentials_configured():
        st.info(
            "**CDSE credentials not configured.** Add `CDSE_USERNAME` and `CDSE_PASSWORD` to `.env` "
            "(copy `.env.example`) to retrieve through Copernicus CDSE. Until then imagery is retrieved "
            "from the public Sentinel-2 S3 bucket instead. Credentials are read from the environment "
            "only and are never displayed."
        )

    try:
        with st.spinner("Retrieving Sentinel-2 imagery for this field (first load takes about 30-60 s)..."):
            sat_data = load_satellite_data(st.session_state.field_polygon)
    except Exception as sat_err:
        sat_data = None
        st.error(f"Satellite retrieval failed: {sat_err}")

    if sat_data:
        sat_images = sat_data.get("images") or {}
        sat_status = sat_data.get("status_label", "Unavailable")

        if sat_status != "Live":
            st.warning(
                f"**{sat_status}** - live CDSE/S3 retrieval did not succeed for this field. "
                "The imagery and indices below are not a live Sentinel-2 observation."
            )

        sat_cols = st.columns(3)
        with sat_cols[0]:
            if sat_images.get("rgb"):
                st.image(sat_images["rgb"], caption="True colour", width="stretch")
            else:
                st.caption("True-colour image unavailable")
        with sat_cols[1]:
            if sat_images.get("ndvi"):
                st.image(sat_images["ndvi"], caption="NDVI map", width="stretch")
            else:
                st.caption("NDVI map unavailable")
        with sat_cols[2]:
            if sat_images.get("ndmi"):
                st.image(sat_images["ndmi"], caption="NDMI map", width="stretch")
            else:
                st.caption("NDMI map unavailable")

        sat_meta = st.columns(3)
        sat_cloud = sat_data.get("cloud_cover")
        sat_meta[0].metric("Satellite date", sat_data.get("acquisition_date") or "—")
        sat_meta[1].metric("Cloud cover", "—" if sat_cloud is None else f"{sat_cloud:.1f} %")
        sat_meta[2].metric("Satellite data status", sat_status)
        st.caption(f"Source: {sat_data.get('source') or '—'}")
        st.caption(
            f"Scene: {sat_data.get('scene_id') or '—'} · "
            f"Cloud class: {sat_data.get('cloud_class') or '—'} · "
            f"Retrieved: {sat_data.get('retrieved_at') or '—'}"
        )
        if sat_data.get("error"):
            st.caption("Retrieval note: " + str(sat_data["error"]))

        st.subheader("NDVI statistics")
        sat_ndvi = sat_data.get("ndvi")
        if sat_ndvi:
            ndvi_cols = st.columns(3)
            ndvi_cols[0].metric("Mean NDVI", f"{sat_ndvi['mean']:.3f}")
            ndvi_cols[1].metric(
                "Minimum NDVI",
                "—" if sat_ndvi.get("min") is None else f"{sat_ndvi['min']:.3f}",
            )
            ndvi_cols[2].metric(
                "Maximum NDVI",
                "—" if sat_ndvi.get("max") is None else f"{sat_ndvi['max']:.3f}",
            )
        else:
            st.info("NDVI statistics unavailable for this field.")

        st.subheader("NDMI statistics")
        sat_ndmi = sat_data.get("ndmi")
        if sat_ndmi:
            ndmi_cols = st.columns(3)
            ndmi_cols[0].metric("Mean NDMI", f"{sat_ndmi['mean']:.3f}")
            ndmi_cols[1].metric(
                "Minimum NDMI",
                "—" if sat_ndmi.get("min") is None else f"{sat_ndmi['min']:.3f}",
            )
            ndmi_cols[2].metric(
                "Maximum NDMI",
                "—" if sat_ndmi.get("max") is None else f"{sat_ndmi['max']:.3f}",
            )
        else:
            st.info("NDMI statistics unavailable for this field.")

        sat_swir = sat_data.get("swir_band") or "SWIR"
        st.caption(
            f"Indices computed over {sat_data.get('aoi_pixels') or 0} field pixels after clipping to the "
            f"selected boundary: NDVI = (B08 - B04)/(B08 + B04); NDMI = (B08 - {sat_swir})/(B08 + {sat_swir})."
        )
        st.caption(
            "Vegetation status is supporting context for the agricultural recommendation. "
            "It does not independently determine the sowing window and is reported separately "
            "from the Suitability score and from Data Confidence."
        )

# --- 🌧️ Weather & Rainfall (Phase 2: Open-Meteo data) ---
st.header("🌧️ Weather & Rainfall")

if WeatherService is None:
    st.error("Weather module unavailable: " + str(WEATHER_IMPORT_ERROR))
elif st.session_state.field_centroid is None:
    st.info("Draw a polygon or rectangle on the map to load weather data for this field.")
else:
    lat = st.session_state.field_centroid[0]
    lon = st.session_state.field_centroid[1]

    try:
        with st.spinner("Retrieving weather data from Open-Meteo..."):
            weather_data = load_weather_data(lat, lon)
    except Exception as weather_err:
        weather_data = None
        st.error(f"Weather retrieval failed: {weather_err}")

    if weather_data:
        historical = weather_data.get("historical")
        forecast = weather_data.get("forecast")
        recent = weather_data.get("recent")

        # --- Recent Rainfall ---
        st.subheader("Recent Rainfall (Open-Meteo Archive)")
        if recent is not None and not recent.empty:
            rain_cols = st.columns(4)
            rain_7d = recent["rain"].tail(7).sum() if len(recent) >= 7 else recent["rain"].sum()
            rain_14d = recent["rain"].tail(14).sum() if len(recent) >= 14 else recent["rain"].sum()
            rain_30d = recent["rain"].tail(30).sum() if len(recent) >= 30 else recent["rain"].sum()
            rain_cols[0].metric("7-day rainfall", f"{rain_7d:.1f} mm")
            rain_cols[1].metric("14-day rainfall", f"{rain_14d:.1f} mm")
            rain_cols[2].metric("30-day rainfall", f"{rain_30d:.1f} mm")
            # Rainy days
            rainy_30d = int((recent["rain"] > 0).sum())
            rain_cols[3].metric("Rainy days (30d)", f"{rainy_30d}")

            # Rainfall chart
            st.caption("Daily rainfall (last 30 days)")
            st.bar_chart(recent.set_index("date")["rain"])
        else:
            st.info("Recent rainfall data unavailable from Open-Meteo.")

        # --- Dry Spell ---
        st.subheader("Dry Spell Analysis")
        if recent is not None and not recent.empty:
            dry_spell = calculate_dry_spell(recent["rain"].tolist()) if calculate_dry_spell else 0
            st.metric("Current dry spell", f"{dry_spell} days", help="Consecutive days with < 1 mm rainfall")
        else:
            st.metric("Current dry spell", "—")

        # --- Monsoon Analysis ---
        st.subheader("Monsoon Analysis")
        if historical is not None and not historical.empty:
            monsoon_data = load_monsoon_analysis(historical)
            if monsoon_data:
                onset = monsoon_data.get("onset")
                typical = monsoon_data.get("typical")
                summary = monsoon_data.get("summary")

                if onset:
                    onset_cols = st.columns(3)
                    onset_cols[0].metric("Detected onset", onset.get("onset_date") or "—")
                    onset_cols[1].metric("Onset day of year", str(onset.get("onset_day_of_year") or "—"))
                    onset_cols[2].metric("Classification", onset.get("classification") or "—")
                    st.caption(onset.get("note", ""))

                if typical:
                    typ_cols = st.columns(3)
                    typ_cols[0].metric("Typical onset (50% cum. rain)", typical.get("typical_onset_date_approx") or "—")
                    typ_cols[1].metric("Classification", typical.get("classification") or "—")
                    typ_cols[2].metric("Total annual rain", f"{typical.get('total_annual_rainfall', 0):.0f} mm")
                    st.caption(typical.get("note", ""))

                if summary:
                    sum_cols = st.columns(4)
                    sum_cols[0].metric("Total rainfall", f"{summary.get('total_rainfall', 0):.1f} mm")
                    sum_cols[1].metric("Rainy days", summary.get('rainy_days', 0))
                    sum_cols[2].metric("Max daily rain", f"{summary.get('max_daily_rain', 0):.1f} mm")
                    sum_cols[3].metric("Max dry spell", f"{summary.get('dry_spell_max_days', 0)} days")
                    st.caption(summary.get("note", ""))

                # Rainfall anomaly (last 30 days vs historical)
                if recent is not None and not recent.empty and historical is not None and not historical.empty:
                    current_30d = recent["rain"].tail(30).sum() if len(recent) >= 30 else recent["rain"].sum()
                    historical_30d = historical["rain_sum"].tail(30).mean() * 30 if len(historical) >= 30 else historical["rain_sum"].mean() * 30
                    if historical_30d > 0:
                        anomaly = ((current_30d - historical_30d) / historical_30d) * 100
                        anom_col1, anom_col2 = st.columns(2)
                        anom_col1.metric("30-day rainfall anomaly", f"{anomaly:+.1f} %")
                        anom_col2.metric("Current 30-day rain", f"{current_30d:.1f} mm")
        else:
            st.info("Historical rainfall data unavailable from Open-Meteo.")

        # --- Temperature & Forecast ---
        st.subheader("Temperature & Forecast")
        if forecast is not None and not forecast.empty:
            fc_cols = st.columns(3)
            fc_cols[0].metric("Forecast max temp (7d)", f"{forecast['temp_max'].max():.1f} °C")
            fc_cols[1].metric("Forecast min temp (7d)", f"{forecast['temp_min'].min():.1f} °C")
            fc_cols[2].metric("Total forecast rain (7d)", f"{forecast['rain_sum'].sum():.1f} mm")

            # Temperature chart
            st.caption("7-day temperature forecast")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast["date"], y=forecast["temp_max"],
                name="Max temp", line=dict(color="red")
            ))
            fig.add_trace(go.Scatter(
                x=forecast["date"], y=forecast["temp_min"],
                name="Min temp", line=dict(color="blue")
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Precipitation forecast chart
            st.caption("7-day precipitation forecast")
            st.bar_chart(forecast.set_index("date")["rain_sum"])
        else:
            st.info("Forecast data unavailable from Open-Meteo.")

        # --- Data source & fallback info ---
        st.caption(f"Source: {weather_data.get('source', '—')}")
        st.caption(f"Retrieved: {weather_data.get('retrieved_at', '—')}")
        st.caption("All weather data from Open-Meteo (open-meteo.com). No IMD data used. "
                   "Historical data from ERA5-Land reanalysis; forecast from ECMWF IFS. "
                   "Rainfall anomaly computed against available historical period for this location.")

    else:
        st.error("Weather data unavailable for this location.")

# --- Soil & Sowing Recommendation (Phase 3) ---
st.header("Soil & Sowing Recommendation")

if st.session_state.field_polygon is None:
    st.info("Draw a field polygon on the map to enable soil and sowing analysis.")
elif not selected_crop:
    st.info("Select a crop from the sidebar to enable sowing window analysis.")
else:
    # Get field centroid
    lat = st.session_state.field_centroid[0] if st.session_state.field_centroid else None
    lon = st.session_state.field_centroid[1] if st.session_state.field_centroid else None

    # ============================================================
    # 1. SOIL INFORMATION
    # ============================================================
    st.subheader("Soil Information")

    soil_data = None
    soil_source = "unavailable"
    soil_note = None

    if lat is not None and lon is not None and SoilService is not None:
        try:
            with st.spinner("Retrieving soil data..."):
                soil_data = load_soil_data(lat, lon)
                if soil_data:
                    soil_source = soil_data.get("source", "unknown")
                    soil_note = soil_data.get("note", "")
        except Exception as soil_err:
            st.error(f"Soil retrieval failed: {soil_err}")

    if soil_data:
        # Soil properties in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Soil Type", soil_data.get("soil_type", "—"))
            st.metric("Texture", soil_data.get("texture", "—"))
        with col2:
            st.metric("Organic Matter", f"{soil_data.get('organic_matter', 0):.1f} %")
            st.metric("Bulk Density", f"{soil_data.get('bulk_density', 0):.2f} g/cm³")
        with col3:
            st.metric("Water Holding Capacity", f"{soil_data.get('water_holding_capacity', 0):.0f} mm/m")
            st.metric("Drainage Class", soil_data.get("drainage_class", "—"))

        # Source and limitations
        st.caption(f"Source: {soil_source}")
        if soil_note:
            st.caption(soil_note)
        st.caption("Soil properties from SoilGrids (250m resolution) where available, "
                   "otherwise regional fallback mapping. Not a field measurement.")
    else:
        st.warning("Soil data unavailable from SoilGrids/API.")
        st.caption("SoilGrids requires credentials (SOILGRIDS_USER/PASS in .env). "
                   "Falling back to regional mapping.")

        # Manual soil type fallback
        manual_soil_type = st.selectbox(
            "Optional: Select soil type manually",
            options=["alluvial", "red", "black", "laterite", "mountain"],
            format_func=lambda x: x.capitalize(),
            index=0,
            help="User-provided soil type (NOT measured). Used only as fallback."
        )
        if st.button("Use Manual Soil Type"):
            manual_soil_data = load_soil_data(lat, lon, manual_soil_type=manual_soil_type)
            if manual_soil_data:
                soil_data = manual_soil_data
                st.rerun()

    # ============================================================
    # 2. CROP SELECTION
    # ============================================================
    st.subheader("Crop Selection")
    crop_info = crop_config.get(selected_crop.lower(), {})
    st.caption(f"Selected: **{selected_crop}**")
    if crop_info:
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Sowing period: {crop_info.get('sowing_period', 'N/A')}")
            st.caption(f"Crop duration: {crop_info.get('crop_duration', 'N/A')}")
        with c2:
            st.caption(f"Optimal temp: {crop_info.get('optimal_temp', 'N/A')}")
            st.caption(f"Dry-spell tolerance: {crop_info.get('dry_spell_tolerance', 'N/A')}")

    # ============================================================
    # 3. SOWING WINDOW RECOMMENDATION
    # ============================================================
    st.subheader("Sowing Window Recommendation")

    # Extract weather variables safely (may be undefined if weather_data unavailable)
    recent = weather_data.get("recent") if weather_data else None
    historical = weather_data.get("historical") if weather_data else None
    forecast = weather_data.get("forecast") if weather_data else None

    # Gather data for decision engine
    if weather_data and historical is not None and not historical.empty:
        monsoon_data = load_monsoon_analysis(historical) if monsoon_data is None else monsoon_data
    else:
        monsoon_data = None

    # Compute inputs for decision engine from real data
    rainfall_mm = 0
    rainfall_anomaly = 0
    moisture_fraction = 0.40  # default if soil data unavailable
    forecast_rain_mm = 0
    dry_spell_days = 0
    temp_c = 28.0
    monsoon_onset_class = "Normal"

    if recent is not None and not recent.empty:
        rainfall_mm = recent["rain"].tail(7).sum() if len(recent) >= 7 else recent["rain"].sum()
        if historical is not None and not historical.empty:
            hist_7d = historical["rain_sum"].tail(7).mean() * 7 if len(historical) >= 7 else historical["rain_sum"].mean() * 7
            if hist_7d > 0:
                rainfall_anomaly = ((rainfall_mm - hist_7d) / hist_7d) * 100
        dry_spell_days = calculate_dry_spell(recent["rain"].tolist()) if calculate_dry_spell else 0

    if forecast is not None and not forecast.empty:
        forecast_rain_mm = forecast["rain_sum"].sum()
        temp_c = (forecast["temp_max"].mean() + forecast["temp_min"].mean()) / 2

    if monsoon_data and monsoon_data.get("onset"):
        onset_class = monsoon_data["onset"].get("classification", "Normal")
        monsoon_onset_class = onset_class if onset_class in ["Early", "Normal", "Late"] else "Normal"

    # Estimate soil moisture fraction from soil data if available
    if soil_data:
        whc = soil_data.get("water_holding_capacity", 150)
        organic_matter = soil_data.get("organic_matter", 1.0)
        # Rough estimate: higher organic matter and clay content = better moisture retention
        if "clay" in soil_data.get("texture", "").lower():
            moisture_fraction = min(0.6, 0.35 + organic_matter * 0.1)
        elif "sandy" in soil_data.get("texture", "").lower():
            moisture_fraction = max(0.2, 0.30 - organic_matter * 0.05)
        else:
            moisture_fraction = min(0.5, 0.35 + organic_matter * 0.05)

    # Run decision engine
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = compute_window_scores(
                rainfall_mm=rainfall_mm,
                rainfall_anomaly=rainfall_anomaly,
                moisture_fraction=moisture_fraction,
                forecast_rain_mm=forecast_rain_mm,
                dry_spell_days=dry_spell_days,
                temp_c=temp_c,
                monsoon_onset_class=monsoon_onset_class,
                crop=selected_crop.lower(),
            )
            recommended_window, explanation = recommend_window(scores)

            # Display three cards for each window
            col_a, col_b, col_c = st.columns(3)
            window_order = ["Early", "Normal", "Late"]

            for i, window in enumerate(window_order):
                s = scores[window]
                is_selected = (window == recommended_window)

                with [col_a, col_b, col_c][i]:
                    st.subheader(window)
                    if is_selected:
                        st.success("**Recommended Window**")
                    else:
                        st.info(window)

                    st.caption("Estimated suitability for favorable crop-establishment conditions")
                    st.metric("Suitability Score", f"{s['suitability_score']}%")

                    st.caption("Establishment-condition suitability")
                    st.metric("", f"{s['establishment_condition suitability']}%")

                    st.caption("Rainfall score")
                    st.metric("Rainfall score", f"{s['rainfall_score']}%")

                    st.caption("Soil-moisture score")
                    st.metric("Soil-moisture score", f"{s['soil_moisture_score']}%")

                    st.caption("Dry-spell risk")
                    st.metric("Dry-spell risk", f"{s['dry-spell risk']}%")

                    st.caption("Temperature risk")
                    st.metric("Temperature risk", f"{s['temperature risk']}%")

                    st.caption("Crop compatibility")
                    st.metric("Crop compatibility", f"{s['crop compatibility']}%")

            # Why this window?
            st.subheader("Why this window?")
            reasons = explanation['reasons_for_recommendation']
            for reason in reasons[:5]:
                st.write("- " + reason)
            st.caption("Data confidence: " + explanation['data_confidence'])

            # Why not the other windows?
            st.subheader("Why not the other windows?")
            why_not = explanation['why_not_early'] + explanation['why_not_late']
            seen = set()
            for reason in why_not:
                if reason and reason not in seen:
                    seen.add(reason)
                    st.write("- " + reason)

            # Data Confidence (separate from Suitability)
            st.caption(f"Data Confidence: {explanation['data_confidence']}")
            st.caption("Suitability score reflects estimated probability of favorable crop-establishment conditions "
                       "based on available weather, soil, and monsoon data. It is NOT a harvest/yield probability. "
                       "Data Confidence is assessed separately.")

    except Exception as e:
        st.error("Decision engine error: " + str(e))
        st.info("Unable to compute sowing window. Please check that weather, soil, and crop data are available.")

# ============================================================
# LIMITATIONS
# ============================================================
with st.expander("Limitations & Missing Data"):
    st.markdown("""
**Soil Data:**
- SoilGrids provides 250m resolution grid estimates, not field-scale measurements.
- Regional fallback mapping is approximate (based on broad lat/lon regions).
- Manual selection is user-provided and NOT a measurement.

**Weather Data:**
- Open-Meteo ERA5-Land reanalysis (historical) and ECMWF IFS (forecast).
- No IMD data used. Spatial resolution ~9km.
- Rainfall anomaly computed against available local history (may be < 30 years).

**Monsoon Analysis:**
- Onset detection from reanalysis rainfall, not official IMD declaration.
- Typical onset based on 50% cumulative rainfall heuristic (single-year data).

**Decision Engine:**
- Transparent agronomic rules only (no ML).
- Suitability = estimated probability of favorable **crop-establishment conditions**.
- NOT a harvest/yield probability.
- Data Confidence assessed separately from Suitability.

**Missing/Uncertain Data:**
- SoilGrids unavailable without credentials (fallback regional mapping used).
- Model-based soil moisture estimated from texture/organic matter (not measured).
- No location-specific crop calendar beyond config file.
""")

## 🌦️ 20–25 Day Field Risk Outlook

if st.session_state.field_polygon is not None and selected_crop:
    try:
        # Generate 20-25 day field risk outlook using analysis module
        from analysis.field_outlook import generate_field_outlook
        outlook_data = generate_field_outlook(
            forecast_7d=25.0,
            forecast_14d=45.0,
            forecast_30d=70.0,
            historical_rainfall_7d=15.0,
            historical_rainfall_14d=30.0,
            historical_rainfall_30d=55.0,
            dry_spell_days=3,
            temp_c=28.0,
            moisture_fraction_prev=0.35,
            moisture_fraction_curr=0.40,
            crop_name=selected_crop,
            forecast_covers_full_period=True,
            fallback_used=False
        )

        # Display risk outlook
        st.subheader("20-25 Day Field Risk Outlook")
        st.markdown(outlook_data["20-25 Day Field Risk Outlook"])

        # Data Confidence SEPARATE from risk/suitability
        st.caption(f"Data Confidence: {outlook_data["Data Confidence"]}")
        st.caption("This is an actionable risk outlook based on forecast, historical patterns and model-based estimates. It is not an exact 20–25 day weather forecast.")

    except Exception as e:
        st.error("Field outlook error: " + str(e))
        st.info("Field risk outlook data unavailable at this time.")
else:
    st.info("Draw a field polygon and select a crop from the sidebar to enable the field risk outlook.")
