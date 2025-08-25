import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import requests
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from branca.colormap import linear
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, Point
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# --- Page Configuration ---
st.set_page_config(
    page_title="Chennai Flood Risk Dashboard",
    page_icon="🌊",
    layout="wide"
)

# --- Configuration & API ---
API_KEY = '4bd56c43f6623156c3831b08e2a13491' # Your OpenWeatherMap API Key
CITY_NAME = 'Chennai'
LAT, LON = 13.0827, 80.2707

# --- File Paths ---
ELEVATION_TIF_PATH = 'dataset/output_hh.tif'
POPULATION_TIF_PATH = 'dataset/chennai_ppp_2020_constrained.tif'
INFRASTRUCTURE_CSV_PATH = 'dataset/chennai_infrastructure.csv'
MODEL_PATH = 'model/rainfall_predictor_model.joblib'
HISTORICAL_DATA_PATH = 'dataset/historical_context_for_prediction.csv'


# --- Helper Functions (Cached for performance) ---

@st.cache_data
def calculate_fvi():
    """
    This version no longer clips the grid over the sea.
    It processes all cells to show risk everywhere, including offshore.
    """
    try:
        with rasterio.open(ELEVATION_TIF_PATH) as src:
            bounds, raster_crs, nodata_value = src.bounds, src.crs, src.nodata
        xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
        grid_size = (xmax - xmin) / 100
        grid_cells = [box(x, y, x + grid_size, y + grid_size) for x in np.arange(xmin, xmax, grid_size) for y in np.arange(ymin, ymax, grid_size)]
        grid_gdf = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=raster_crs)
    except Exception as e:
        st.error(f"Grid creation failed: {e}")
        return None

    # Calculate elevation, filling sea-level cells with 0 instead of NaN
    with rasterio.open(ELEVATION_TIF_PATH) as src:
        elevations = []
        for geom in grid_gdf['geometry']:
            try:
                out_image, _ = mask(src, [geom], crop=True)
                if nodata_value is not None:
                    out_image[out_image == nodata_value] = np.nan
                mean_elev = np.nanmean(out_image)
                elevations.append(mean_elev if not np.isnan(mean_elev) else 0)
            except (ValueError, IndexError):
                elevations.append(0)
    grid_gdf['avg_elevation'] = elevations
    grid_gdf.dropna(subset=['avg_elevation'], inplace=True)

    with rasterio.open(POPULATION_TIF_PATH) as src:
        grid_gdf_pop = grid_gdf.to_crs(src.crs)
        populations = []
        for geom in grid_gdf_pop['geometry']:
            try:
                out_image, _ = mask(src, [geom], crop=True, nodata=0)
                out_image[out_image < 0] = 0
                populations.append(np.nansum(out_image))
            except (ValueError, IndexError):
                populations.append(0)
    grid_gdf['population'] = populations
    grid_gdf.dropna(subset=['population'], inplace=True)
    
    infra_df = pd.read_csv(INFRASTRUCTURE_CSV_PATH)
    infra_gdf = gpd.GeoDataFrame(infra_df, geometry=gpd.points_from_xy(infra_df.longitude, infra_df.latitude), crs="EPSG:4326")
    grid_gdf_proj = grid_gdf.to_crs("EPSG:32644")
    infra_gdf_proj = infra_gdf.to_crs("EPSG:32644")
    grid_gdf['dist_to_infra'] = grid_gdf_proj.geometry.apply(lambda g: infra_gdf_proj.distance(g).min())
    
    scaler = RobustScaler()
    def elevation_to_risk(elev_m):
        k, x0 = 2.5, 1.0
        return float(np.clip(1.0 / (1.0 + np.exp(k * (elev_m - x0))), 0.0, 0.95))

    grid_gdf['elevation_norm'] = grid_gdf['avg_elevation'].apply(elevation_to_risk)
    grid_gdf['population_norm'] = scaler.fit_transform(grid_gdf[['population']])
    grid_gdf['infra_norm'] = scaler.fit_transform(grid_gdf[['dist_to_infra']])
    
    weights = {'exposure': 0.7, 'sensitivity': 0.3}
    exposure_score = grid_gdf['elevation_norm']
    sensitivity_score = (grid_gdf['population_norm'] * 0.4) + (grid_gdf['infra_norm'] * 0.6)
    grid_gdf['FVI'] = (exposure_score * weights['exposure']) + (sensitivity_score * weights['sensitivity'])
    grid_gdf['id'] = grid_gdf.index.astype(str)
    return grid_gdf

def create_map(gdf, infra_df, risk_column, legend_name, searched_location=None):
    if searched_location:
        map_center, zoom_start = [searched_location.latitude, searched_location.longitude], 15
    else:
        map_center, zoom_start = [13.0827, 80.2707], 11
    
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=None)
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('Esri.WorldImagery', name='Satellite View').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)

    colormap = linear.YlOrRd_09.scale(vmin=0, vmax=1)
    colormap.caption = legend_name
    m.add_child(colormap)

    ### START OF MODIFICATION ###
    # Create new columns with formatted strings for display purposes
    gdf['FVI_formatted'] = gdf['FVI'].map('{:.3f}'.format)
    gdf['population_formatted'] = gdf['population'].map('{:,.0f}'.format)
    gdf['elevation_formatted'] = gdf['avg_elevation'].map('{:.2f} m'.format)
    # The risk_column variable will be 'real_time_risk_norm' or 'forecasted_risk_norm'
    gdf['risk_score_formatted'] = gdf[risk_column].map('{:.3f}'.format)

    # Define the fields and aliases to use for the tooltip and popup
    tooltip_fields = ["FVI_formatted"]
    tooltip_aliases = ["FVI Score:"]
    
    popup_fields = ["FVI_formatted", "population_formatted", "elevation_formatted", "risk_score_formatted"]
    popup_aliases = ["Flood Vulnerability Index:", "Est. Population:", "Avg. Elevation:", "Current Risk Score:"]
    ### END OF MODIFICATION ###

    risk_layer = folium.FeatureGroup(name='Flood Risk Heatmap', show=True); m.add_child(risk_layer)
    infra_layer = folium.FeatureGroup(name='Critical Infrastructure', show=True); m.add_child(infra_layer)

    folium.GeoJson(
        gdf,
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties'][risk_column]),
            'color': 'transparent',
            'weight': 0,
            'fillOpacity': 0.7
        },
        ### MODIFIED ###
        # Use the new formatted fields and aliases
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, style="background-color: white; color: #333333;"),
        popup=folium.GeoJsonPopup(fields=popup_fields, aliases=popup_aliases, localize=True)
        ### END OF MODIFICATION ###
    ).add_to(risk_layer)

    for _, row in infra_df.iterrows():
        folium.Marker(location=[row['latitude'], row['longitude']], popup=f"<b>{row['name']}</b><br>Type: {row['type']}", icon=folium.Icon(color='blue', icon='info-sign')).add_to(infra_layer)
    if searched_location:
        folium.Marker(location=[searched_location.latitude, searched_location.longitude], popup=f"<b>Searched Location:</b><br>{searched_location.address}", icon=folium.Icon(color='green', icon='search')).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

@st.cache_resource
def load_model_and_history():
    model = joblib.load(MODEL_PATH)
    historical_data = pd.read_csv(HISTORICAL_DATA_PATH)
    historical_data['Timestamp'] = pd.to_datetime(historical_data['Timestamp'])
    return model, historical_data

@st.cache_data(ttl=600)
def get_weather_data(api_key):
    current_weather = None; forecast_df = None
    try:
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY_NAME}&appid={api_key}&units=metric"; response_current = requests.get(current_url); response_current.raise_for_status(); data_current = response_current.json()
        current_weather = {'Temperature (°C)': data_current['main'].get('temp', 'N/A'), 'Feels Like (°C)': data_current['main'].get('feels_like', 'N/A'), 'Humidity (%)': data_current['main'].get('humidity', 'N/A'), 'Conditions': data_current['weather'][0].get('description', 'N/A').title(), 'Wind Speed (kph)': data_current['wind'].get('speed', 0) * 3.6}
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={api_key}&units=metric"; response_forecast = requests.get(forecast_url); response_forecast.raise_for_status(); data_forecast = response_forecast.json()
        forecast_list = [{'Timestamp': pd.to_datetime(item['dt_txt']), 'temperature_c': item['main'].get('temp', 0), 'humidity_percent': item['main'].get('humidity', 0), 'wind_speed_kph': item['wind'].get('speed', 0) * 3.6, 'rainfall_mm': item.get('rain', {}).get('3h', 0) / 3} for item in data_forecast['list'][:8]]
        forecast_df = pd.DataFrame(forecast_list)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data from API: {e}")
    return current_weather, forecast_df

def engineer_features_for_prediction(historical_df, new_data_df):
    historical_context = historical_df.tail(72); combined_df = pd.concat([historical_context, new_data_df], ignore_index=True); combined_df.ffill(inplace=True)
    combined_df['hour_of_day'] = combined_df['Timestamp'].dt.hour; combined_df['day_of_year'] = combined_df['Timestamp'].dt.dayofyear; combined_df['month'] = combined_df['Timestamp'].dt.month; combined_df['day_of_week'] = combined_df['Timestamp'].dt.dayofweek
    for lag in [1, 3, 6, 12, 24]:
        combined_df[f'temp_lag_{lag}h'] = combined_df['temperature_c'].shift(lag)
        if 'humidity_percent' in combined_df.columns: combined_df[f'humidity_lag_{lag}h'] = combined_df['humidity_percent'].shift(lag)
    for window in [3, 6, 12, 24, 72]:
        combined_df[f'rainfall_sum_{window}h'] = combined_df['rainfall_mm'].rolling(window=window, min_periods=1).sum()
        combined_df[f'temp_mean_{window}h'] = combined_df['temperature_c'].rolling(window=window, min_periods=1).mean()
        if 'wind_speed_kph' in combined_df.columns: combined_df[f'wind_max_{window}h'] = combined_df['wind_speed_kph'].rolling(window=window, min_periods=1).max()
    return combined_df.tail(len(new_data_df))

# --- Main Application ---
st.title("🌊 Chennai Interactive Flood Risk Dashboard")

if 'searched_location' not in st.session_state:
    st.session_state.searched_location = None

with st.spinner("Loading assets and calculating base vulnerability..."):
    base_fvi_gdf = calculate_fvi()
    if base_fvi_gdf is not None:
        infra_df = pd.read_csv(INFRASTRUCTURE_CSV_PATH)
        model, historical_data = load_model_and_history()
    else:
        st.error("Could not generate the base Flood Vulnerability Index. The application cannot proceed."); st.stop()
st.info("Newer Gen Model is on the developement")
st.sidebar.header("Dashboard Options")
view_selection = st.sidebar.radio("Select View:", ("Current Risk", "24-Hour Forecasted Risk"))
st.sidebar.header("Risk Simulator")
is_simulating = st.sidebar.checkbox("Enable 'What-If' Scenario")
simulated_rainfall = 0
if is_simulating:
    simulated_rainfall = st.sidebar.slider("Simulate Rainfall (mm in next 3h)", 0.0, 50.0, 10.0, 0.5)
st.sidebar.header("Location Risk Lookup")
address_input = st.sidebar.text_input("Enter an address in Chennai", "")
if st.sidebar.button("Search"):
    if address_input:
        geolocator = Nominatim(user_agent="chennai_flood_risk_app")
        try:
            st.session_state.searched_location = geolocator.geocode(f"{address_input}, Chennai")
            if st.session_state.searched_location is None: st.sidebar.error("Address not found. Please try again.")
        except (GeocoderTimedOut, GeocoderUnavailable):
            st.sidebar.error("Geocoding service is unavailable. Please try again later.")
    else:
        st.sidebar.warning("Please enter an address.")

if base_fvi_gdf is not None and model is not None:
    MAX_HAZARD_SCORE = 3
    if not base_fvi_gdf['FVI'].empty:
        MAX_POSSIBLE_RISK = base_fvi_gdf['FVI'].max() * MAX_HAZARD_SCORE
    else:
        MAX_POSSIBLE_RISK = 3.0

    if st.session_state.searched_location:
        point = Point(st.session_state.searched_location.longitude, st.session_state.searched_location.latitude)
        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs="EPSG:4326").to_crs(base_fvi_gdf.crs)
        try:
            cell_data = gpd.sjoin(point_gdf, base_fvi_gdf, how="inner", predicate='within')
            if not cell_data.empty:
                st.sidebar.subheader("Risk at Searched Location"); fvi_score = cell_data['FVI'].iloc[0]
                st.sidebar.metric("Flood Vulnerability Index (FVI)", f"{fvi_score:.3f}")
                st.sidebar.markdown(f"**Est. Population:** {int(cell_data['population'].iloc[0])}")
                st.sidebar.markdown(f"**Avg. Elevation:** {cell_data['avg_elevation'].iloc[0]:.2f} m")
            else:
                st.sidebar.warning("Location is outside the analysis grid.")
        except Exception:
            st.sidebar.error("Could not determine risk for the location.")

    if view_selection == "Current Risk":
        st.header("Current Flood Risk Assessment")
        current_weather, _ = get_weather_data(API_KEY)
        if current_weather:
            if is_simulating:
                predicted_rainfall = simulated_rainfall
                st.info(f" simulating risk for **{predicted_rainfall:.2f} mm** of rainfall.")
            else:
                current_weather_df = pd.DataFrame([{'Timestamp': datetime.now(),'temperature_c': current_weather.get('Temperature (°C)',0),'humidity_percent': current_weather.get('Humidity (%)',0),'wind_speed_kph': current_weather.get('Wind Speed (kph)',0),'rainfall_mm': 0}])
                features_df = engineer_features_for_prediction(historical_data, current_weather_df)
                predicted_rainfall = max(0, model.predict(features_df[model.get_booster().feature_names])[0])
            
            if predicted_rainfall <= 5: hazard_score = 1
            elif 5 < predicted_rainfall <= 15: hazard_score = 2
            else: hazard_score = 3
            
            current_risk_gdf = base_fvi_gdf.copy()
            current_risk_gdf['real_time_risk'] = current_risk_gdf['FVI'] * (hazard_score**1.5)
            current_risk_gdf['real_time_risk_norm'] = (current_risk_gdf['real_time_risk'] / MAX_POSSIBLE_RISK).clip(0, 1)

            tab1, tab2, tab3 = st.tabs(["Risk Map", "Current Weather", "At-Risk Infrastructure"])
            with tab1:
                col1, col2 = st.columns(2)
                col1.metric("Rainfall (Next 3 Hours)", f"{predicted_rainfall:.2f} mm")
                col2.metric("Calculated Hazard Level", {1: "Low", 2: "Medium", 3: "High"}.get(hazard_score, "Unknown"))
                m = create_map(current_risk_gdf, infra_df, 'real_time_risk_norm', 'Normalized Real-Time Flood Risk', st.session_state.searched_location)
                st_folium(m, width=1200, height=600, returned_objects=[])
            with tab2:
                st.subheader("Current Weather Conditions"); st.json(current_weather)
            with tab3:
                st.subheader("Prioritized At-Risk Infrastructure")
                infra_gdf = gpd.GeoDataFrame(infra_df, geometry=gpd.points_from_xy(infra_df.longitude, infra_df.latitude), crs="EPSG:4326").to_crs(current_risk_gdf.crs)
                at_risk_infra = gpd.sjoin(infra_gdf, current_risk_gdf, how="inner", predicate='within')
                priority_list = at_risk_infra.sort_values(by='real_time_risk_norm', ascending=False)
                st.dataframe(priority_list[['name', 'type', 'real_time_risk_norm']].rename(columns={'real_time_risk_norm': 'Risk Score'}).reset_index(drop=True))

    elif view_selection == "24-Hour Forecasted Risk":
        st.header("24-Hour Forecasted Flood Risk")
        _, forecast_df = get_weather_data(API_KEY)
        if forecast_df is not None and not forecast_df.empty:
            features_for_forecast = engineer_features_for_prediction(historical_data, forecast_df)
            forecast_df['predicted_rainfall'] = model.predict(features_for_forecast[model.get_booster().feature_names])
            forecast_df['predicted_rainfall'] = forecast_df['predicted_rainfall'].apply(lambda x: max(0, x))
            max_predicted_rain = forecast_df['predicted_rainfall'].sum()
            
            if max_predicted_rain <= 5: hazard_score = 1
            elif 5 < max_predicted_rain <= 15: hazard_score = 2
            else: hazard_score = 3
            
            forecast_risk_gdf = base_fvi_gdf.copy()
            forecast_risk_gdf['forecasted_risk'] = forecast_risk_gdf['FVI'] * (hazard_score**1.5)
            forecast_risk_gdf['forecasted_risk_norm'] = (forecast_risk_gdf['forecasted_risk'] / MAX_POSSIBLE_RISK).clip(0, 1)
            
            col1, col2 = st.columns(2)
            col1.metric("Max Predicted Rainfall (Next 24h)", f"{max_predicted_rain:.2f} mm")
            col2.metric("Peak Hazard Level", {1: "Low", 2: "Medium", 3: "High"}.get(hazard_score, "Unknown"))
            m_forecast = create_map(forecast_risk_gdf, infra_df, 'forecasted_risk_norm', 'Normalized Forecasted Flood Risk', st.session_state.searched_location)
            st_folium(m_forecast, width=1200, height=600, returned_objects=[])
            st.subheader("24-Hour Rainfall Forecast Visualization")

            st.bar_chart(forecast_df.set_index('Timestamp')['predicted_rainfall'])
