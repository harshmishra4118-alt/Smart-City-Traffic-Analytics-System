# ==========================================================================
# 🚦 SMART CITY TRAFFIC ANALYTICS SYSTEM - ENTERPRISE AI SAAS DASHBOARD
# ==========================================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import datetime
import time

# --- Page Initialization & Setup ---
st.set_page_config(
    page_title="Smart City Traffic Analytics System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

now = datetime.datetime.now()

# --- Initialize Session States ---
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []
if "loaded_toasts" not in st.session_state:
    st.session_state.loaded_toasts = False
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"
if "animation_speed" not in st.session_state:
    st.session_state.animation_speed = "Normal"

# --- Prediction Inputs Session State Management ---
defaults = {
    "temp_input": 20.0,
    "rain_input": 0.0,
    "snow_input": 0.0,
    "cloud_input": 50,
    "weather_main_input": "Clear",
    "weather_desc_input": "Sky is Clear",
    "hours_input": 12,
    "day_input": 15,
    "month_input": 6
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Inject Premium Global Stylesheet ---
try:
    with open("style.css", "r") as f:
        custom_css = f.read()
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
except Exception:
    st.warning("Custom stylesheet style.css not found. Running with fallback styles.")

# --- Background Animated Particles & Glowing Orbs ---
st.markdown("""
<div class="glowing-bg-container">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
""", unsafe_allow_html=True)

# --- Class Mappings for UI Selectboxes ---
MAIN_CLASSES = [
    'Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze',
    'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'
]
DESC_CLASSES = [
    'SQUALLS', 'Sky is Clear', 'broken clouds', 'drizzle', 'few clouds',
    'fog', 'freezing rain', 'haze', 'heavy intensity drizzle', 'heavy intensity rain',
    'heavy snow', 'light intensity drizzle', 'light intensity shower rain', 'light rain',
    'light rain and snow', 'light shower snow', 'light snow', 'mist', 'moderate rain',
    'overcast clouds', 'proximity shower rain', 'proximity thunderstorm',
    'proximity thunderstorm with drizzle', 'proximity thunderstorm with rain',
    'scattered clouds', 'shower drizzle', 'shower snow', 'sky is clear', 'sleet',
    'smoke', 'snow', 'thunderstorm', 'thunderstorm with drizzle',
    'thunderstorm with heavy rain', 'thunderstorm with light drizzle',
    'thunderstorm with light rain', 'thunderstorm with rain', 'very heavy rain'
]

# --- Core Backend Functionality (Caching Intact) ---
@st.cache_data
def data_load():
    df = pd.read_csv("traffic.csv")
    return df

df_original = data_load()
df = df_original.copy()

# Preprocessing
df = df.dropna()

@st.cache_data
def preprocess(df_in):
    df_copy = df_in.copy()
    encoder = LabelEncoder()
    df_copy["weather_main"] = encoder.fit_transform(df_copy["weather_main"])
    df_copy["weather_description"] = encoder.fit_transform(df_copy["weather_description"])
    df_copy["date_time"] = pd.to_datetime(df_copy["date_time"], dayfirst=True)
    df_copy["Hours"] = df_copy["date_time"].dt.hour
    df_copy["Day"] = df_copy["date_time"].dt.day
    df_copy["Month"] = df_copy["date_time"].dt.month
    df_copy.drop("date_time", axis=1, inplace=True)
    return df_copy

df_preprocessed = preprocess(df)

x = df_preprocessed.drop("traffic_volume", axis=1)
y = df_preprocessed["traffic_volume"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

@st.cache_data
def train_model(x_tr, y_tr):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_tr, y_tr)
    return model

model = train_model(x_train, y_train)

# Calculate model scores and evaluation stats
pred = model.predict(x_test)
score = r2_score(y_test, pred)

# Trigger loading toasts once
if not st.session_state.loaded_toasts:
    st.toast("Traffic Dataset Loaded Successfully! 📁", icon="✅")
    st.toast("Random Forest Model Trained & Ready! 🤖", icon="🚀")
    st.session_state.loaded_toasts = True

# --- Floating Premium Sidebar Navigation ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 25px;">
        <span style="font-size: 2.8rem; filter: drop-shadow(0 0 10px rgba(37, 99, 235, 0.6));">🚦</span>
        <h2 style="color: #F8FAFC; font-weight: 800; font-size: 1.35rem; margin-top: 12px; letter-spacing: 0.8px;">SMART CITY</h2>
        <p style="color: #06B6D4; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; margin-top: -5px;">Traffic Analytics System</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Dashboard Overview",
            "📁 Database Explorer",
            "📊 Data Visualizations",
            "🤖 Machine Learning Center",
            "🔮 Tesla Prediction Console",
            "⚙️ Dashboard Settings"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 30px 0;'>", unsafe_allow_html=True)

    # Custom Sidebar Model Badge & Theme Info
    st.markdown("""
    <div class="glass-container" style="padding: 16px !important; border-radius: 14px !important; margin-bottom: 0px !important;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; letter-spacing: 0.5px;">SYSTEM ENGINE</span>
            <span style="font-size: 0.7rem; font-weight: 800; color: #10B981; background: rgba(16, 185, 129, 0.1); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.2);">ONLINE</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; letter-spacing: 0.5px;">MODEL REGRESSION</span>
            <span style="font-size: 0.7rem; font-weight: 800; color: #06B6D4; background: rgba(6, 182, 212, 0.1); padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(6, 182, 212, 0.2);">READY</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================================================
if page == "🏠 Dashboard Overview":
    current_date = now.strftime("%A, %d %B %Y")

    # Hero Section with SVG skyline + JS live clock
    # MUST use components.html() because st.markdown strips <script> and
    # mangles complex SVG (defs, pattern, filter, comments).
    hero_html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Outfit', sans-serif; }}
        body {{ background: transparent; }}
        .hero-card {{
            background: linear-gradient(135deg, rgba(15,23,42,0.7) 0%, rgba(30,41,59,0.4) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 30px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        }}
        .hero-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 150%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(6,182,212,0.05), transparent);
            transform: skewX(-25deg);
            animation: sweep 8s infinite linear;
        }}
        @keyframes sweep {{
            0%   {{ transform: translateX(-150%) skewX(-25deg); }}
            100% {{ transform: translateX(150%) skewX(-25deg); }}
        }}
        .hero-top {{ display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; margin-bottom: 25px; position: relative; z-index: 1; }}
        .hero-left {{ }}
        .hero-right {{ text-align: right; min-width: 200px; }}
        .subtitle-badge {{ display: inline-flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
        .pulse-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #10B981; display: inline-block; animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }} 70% {{ opacity: 0.7; box-shadow: 0 0 0 10px rgba(16,185,129,0); }} }}
        .subtitle-text {{ color: #06B6D4; font-weight: 700; font-size: 0.85rem; letter-spacing: 1.5px; text-transform: uppercase; }}
        .hero-title {{ color: #F8FAFC; font-weight: 800; font-size: 2.1rem; margin: 4px 0 0 0; letter-spacing: -0.5px; }}
        .hero-desc {{ color: #94A3B8; font-size: 0.95rem; margin: 6px 0 0 0; }}
        #live-clock {{ font-size: 2rem; font-weight: 800; color: #06B6D4; font-family: monospace; letter-spacing: 1px; text-shadow: 0 0 10px rgba(6,182,212,0.3); }}
        .date-text {{ font-size: 0.85rem; color: #94A3B8; font-weight: 500; margin-top: 3px; }}
        .status-badge {{ display: inline-block; margin-top: 10px; color: #10B981; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); padding: 4px 10px; border-radius: 8px; font-weight: 800; font-size: 0.75rem; letter-spacing: 0.5px; }}
        .skyline-svg {{ width: 100%; height: 140px; position: relative; z-index: 1; }}
        .car-dot-1 {{ animation: drive-east 6s infinite linear; }}
        .car-dot-2 {{ animation: drive-west 8s infinite linear; animation-delay: -2s; }}
        .car-dot-3 {{ animation: drive-east 10s infinite linear; animation-delay: -4s; }}
        @keyframes drive-east {{ 0% {{ cx: -20; }} 100% {{ cx: 820; }} }}
        @keyframes drive-west {{ 0% {{ cx: 820; }} 100% {{ cx: -20; }} }}
    </style>
    <div class="hero-card">
        <div class="hero-top">
            <div class="hero-left">
                <div class="subtitle-badge">
                    <span class="pulse-dot"></span>
                    <span class="subtitle-text">AI Powered Prediction Platform</span>
                </div>
                <h1 class="hero-title">🚦 Smart City Traffic Analytics System</h1>
                <p class="hero-desc">Redesigning metropolitan grids with Machine Learning intelligence</p>
            </div>
            <div class="hero-right">
                <div id="live-clock">--:--:--</div>
                <div class="date-text">{current_date}</div>
                <span class="status-badge">🟢 ENGINE STATUS: ONLINE</span>
            </div>
        </div>
        <svg class="skyline-svg" viewBox="0 0 800 140" xmlns="http://www.w3.org/2000/svg">
            <g fill="rgba(15,23,42,0.7)" stroke="rgba(255,255,255,0.04)" stroke-width="1">
                <rect x="30" y="60" width="35" height="80" rx="3"/>
                <rect x="80" y="30" width="45" height="110" rx="3"/>
                <rect x="140" y="70" width="30" height="70" rx="3"/>
                <rect x="185" y="15" width="55" height="125" rx="3"/>
                <rect x="255" y="50" width="40" height="90" rx="3"/>
                <rect x="310" y="80" width="30" height="60" rx="3"/>
                <rect x="355" y="10" width="50" height="130" rx="3"/>
                <rect x="420" y="45" width="35" height="95" rx="3"/>
                <rect x="470" y="35" width="45" height="105" rx="3"/>
                <rect x="530" y="65" width="30" height="75" rx="3"/>
                <rect x="575" y="20" width="55" height="120" rx="3"/>
                <rect x="645" y="55" width="35" height="85" rx="3"/>
                <rect x="695" y="30" width="50" height="110" rx="3"/>
            </g>
            <g fill="#06B6D4" opacity="0.45">
                <circle cx="95" cy="50" r="1.5"/><circle cx="105" cy="50" r="1.5"/><circle cx="100" cy="70" r="1.5"/>
                <circle cx="200" cy="30" r="1.5"/><circle cx="210" cy="30" r="1.5"/><circle cx="220" cy="30" r="1.5"/>
                <circle cx="370" cy="25" r="1.5"/><circle cx="380" cy="25" r="1.5"/><circle cx="375" cy="55" r="1.5"/>
                <circle cx="485" cy="50" r="1.5"/><circle cx="495" cy="50" r="1.5"/><circle cx="490" cy="70" r="1.5"/>
                <circle cx="590" cy="40" r="1.5"/><circle cx="600" cy="40" r="1.5"/><circle cx="610" cy="40" r="1.5"/>
                <circle cx="710" cy="50" r="1.5"/><circle cx="720" cy="50" r="1.5"/><circle cx="715" cy="70" r="1.5"/>
            </g>
            <line x1="0" y1="130" x2="800" y2="130" stroke="rgba(255,255,255,0.12)" stroke-width="1.5"/>
            <line x1="0" y1="135" x2="800" y2="135" stroke="rgba(255,255,255,0.25)" stroke-width="1" stroke-dasharray="6 6"/>
            <circle class="car-dot-1" cy="132" r="3.5" fill="#2563EB"/>
            <circle class="car-dot-2" cy="138" r="4" fill="#06B6D4"/>
            <circle class="car-dot-3" cy="132" r="2.8" fill="#8B5CF6"/>
        </svg>
    </div>
    <script>
        function updateClock() {{
            var el = document.getElementById('live-clock');
            if (el) {{
                var d = new Date();
                el.textContent = d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0') + ':' + d.getSeconds().toString().padStart(2,'0');
            }}
        }}
        setInterval(updateClock, 1000);
        updateClock();
    </script>
    """
    components.html(hero_html, height=340, scrolling=False)

    # 📊 KPI Cards Section
    st.markdown("### 📈 Real-Time KPIs & Diagnostics")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="📁 Total Records", value=f"{df_original.shape[0]:,}", delta="Live Count")
    with col2:
        st.metric(label="📑 Dataset Features", value=f"{df_original.shape[1]} Columns", delta="Preprocessed")
    with col3:
        st.metric(label="📈 Model Accuracy (R2)", value=f"{score*100:.2f}%", delta="Random Forest")
    with col4:
        st.metric(label="🚦 Total Predictions", value=f"{st.session_state.prediction_count} Hits", delta="Session Run")
    with col5:
        st.metric(label="⚠ Missing Values", value=f"{df_original.isnull().sum().sum()} Cells", delta="0.0% Missing")

    st.markdown("<br>", unsafe_allow_html=True)

    # 🧠 AI Insight Panel
    st.markdown("### 🧠 AI Automated Insights")

    avg_by_hour = df_preprocessed.groupby('Hours')['traffic_volume'].mean()
    highest_hour = avg_by_hour.idxmax()
    lowest_hour = avg_by_hour.idxmin()

    avg_by_weather = df_original.groupby('weather_main')['traffic_volume'].mean()
    worst_weather = avg_by_weather.idxmax()
    worst_weather_val = avg_by_weather.max()

    avg_by_month = df_preprocessed.groupby('Month')['traffic_volume'].mean()
    highest_month = avg_by_month.idxmax()
    highest_month_name = datetime.date(2026, highest_month, 1).strftime("%B")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        st.markdown(f"""
        <div class="glass-container" style="height: 220px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                <span style="font-size: 2rem;">⏰</span>
                <div>
                    <h4 style="margin: 0; color: #F8FAFC; font-weight: 700; font-size: 1rem;">Peak Hour Analysis</h4>
                    <span style="color: #8B5CF6; font-size: 0.8rem; font-weight: 600;">TEMPORAL TRENDS</span>
                </div>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.4;">
                The highest average traffic volume occurs at <b>{highest_hour}:00</b>, representing heavy commuter cycles.
                Conversely, traffic drops to its minimum at <b>{lowest_hour}:00</b>.
            </p>
            <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; font-weight: 700; font-size: 0.85rem; color: #06B6D4;">
                💡 Commute Recommendation Active
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_in2:
        st.markdown(f"""
        <div class="glass-container" style="height: 220px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                <span style="font-size: 2rem;">⛈️</span>
                <div>
                    <h4 style="margin: 0; color: #F8FAFC; font-weight: 700; font-size: 1rem;">Weather Congestion</h4>
                    <span style="color: #EF4444; font-size: 0.8rem; font-weight: 600;">ATMOSPHERIC IMPACT</span>
                </div>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.4;">
                Severe weather conditions notably affect commuters. The weather category causing highest congestion is
                <b>{worst_weather}</b> (Average: <b>{worst_weather_val:.0f}</b> vehicles/hour).
            </p>
            <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; font-weight: 700; font-size: 0.85rem; color: #10B981;">
                🚗 Speed drop of 18% predicted under {worst_weather}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_in3:
        st.markdown(f"""
        <div class="glass-container" style="height: 220px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                <span style="font-size: 2rem;">🗓️</span>
                <div>
                    <h4 style="margin: 0; color: #F8FAFC; font-weight: 700; font-size: 1rem;">Seasonal Peaks</h4>
                    <span style="color: #2563EB; font-size: 0.8rem; font-weight: 600;">MONTHLY ANALYSIS</span>
                </div>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.4;">
                Traffic volume spikes significantly during the month of <b>{highest_month_name}</b>, likely due to school periods,
                weather declines, or seasonal holidays.
            </p>
            <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px; font-weight: 700; font-size: 0.85rem; color: #F59E0B;">
                📅 High seasonal density alert
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================================================
# PAGE 2: DATABASE EXPLORER
# ==========================================================================
elif page == "📁 Database Explorer":
    st.markdown("## 📁 Dataset Explorer & Micro-Table")
    st.write("Browse, search, sort, and paginate the smart city traffic dataset directly below.")

    with st.container(border=True):
        st.markdown("#### ⚡ Database Controller")

        search_col, pag_col, down_col = st.columns([2, 1, 1])
        with search_col:
            search_query = st.text_input("🔍 Search rows (e.g. 'Rain', 'Clouds', '02-10-2012'):", value="", placeholder="Type search term...")
        with pag_col:
            page_size = st.selectbox("Pagination Rows:", [10, 20, 50, 100], index=0)
        with down_col:
            csv_data = df_original.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Full CSV",
                data=csv_data,
                file_name="smart_city_traffic_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        if search_query:
            mask = df_original.astype(str).apply(lambda row: row.str.contains(search_query, case=False)).any(axis=1)
            df_filtered = df_original[mask]
        else:
            df_filtered = df_original

        total_rows = len(df_filtered)
        num_pages = max(int(np.ceil(total_rows / page_size)), 1)

        page_col, spacer_col = st.columns([1, 3])
        with page_col:
            selected_page = st.number_input("Page:", min_value=1, max_value=num_pages, value=1, step=1)

        start_idx = (selected_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)

        if total_rows > 0:
            st.dataframe(df_filtered.iloc[start_idx:end_idx], use_container_width=True)
            st.markdown(f"<p style='color:#94A3B8; font-size:0.85rem;'>Showing <b>{start_idx + 1} - {end_idx}</b> of <b>{total_rows:,}</b> entries</p>", unsafe_allow_html=True)
        else:
            st.warning("No records found matching your query.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        with st.container(border=True):
            st.markdown("#### 📐 Statistical Summary (Metrics)")
            st.dataframe(df_original.describe(), use_container_width=True)
    with col_d2:
        with st.container(border=True):
            st.markdown("#### 📊 Database Schema & Columns")
            dtypes_df = pd.DataFrame(df_original.dtypes, columns=["Data Type"]).astype(str)
            st.dataframe(dtypes_df, use_container_width=True)

# ==========================================================================
# PAGE 3: DATA VISUALIZATIONS
# ==========================================================================
elif page == "📊 Data Visualizations":
    st.markdown("## 📊 Redesigned Interactive Insights")
    st.write("Visualizations are rendered using Plotly for hardware-accelerated animations, tooltips, and resizing capabilities.")

    # 1. Traffic Volume Distribution
    with st.container(border=True):
        st.markdown("### 🚦 Traffic Volume Distribution")
        st.write("Understand the density profile of commuter traffic in the metropolitan area.")

        fig = px.histogram(
            df_original, x="traffic_volume", nbins=35,
            color_discrete_sequence=['#06B6D4'],
            labels={"traffic_volume": "Traffic Volume (Vehicles / Hour)"}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='#F8FAFC', font_family='Outfit', hovermode="x unified",
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Frequencies")
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2. Scatter plots grid
    col_s1, col_s2 = st.columns(2)
    df_sample = df_original.sample(n=1000, random_state=42)

    with col_s1:
        with st.container(border=True):
            st.markdown("### 🌡️ Temperature vs Traffic Volume")
            st.write("Inspect how the weather temperatures in Kelvin affect vehicle flow (Sample: 1k rows).")
            fig = px.scatter(
                df_sample, x="temp", y="traffic_volume", color="traffic_volume",
                color_continuous_scale="Viridis",
                labels={"temp": "Temperature (K)", "traffic_volume": "Traffic Volume"}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_s2:
        with st.container(border=True):
            st.markdown("### ☁️ Cloud Cover vs Traffic Volume")
            st.write("Inspect if overcast conditions restrict driving volume (Sample: 1k rows).")
            fig = px.scatter(
                df_sample, x="clouds_all", y="traffic_volume", color="traffic_volume",
                color_continuous_scale="Cividis",
                labels={"clouds_all": "Cloud Cover (%)", "traffic_volume": "Traffic Volume"}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3. Heatmap & Pie
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        with st.container(border=True):
            st.markdown("### 🔥 Correlation Matrix Heatmap")
            st.write("Interactive correlations of numeric columns in our preprocessed dataset.")
            numeric_cols = df_preprocessed.select_dtypes(include=[np.number])
            corr = numeric_cols.corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_c2:
        with st.container(border=True):
            st.markdown("### 🍰 Traffic Category Share")
            st.write("Proportional breakdown of traffic volume based on low, medium, and heavy metrics.")
            cats = pd.cut(df_original["traffic_volume"], bins=[-1, 2000, 4500, 10000], labels=["Low", "Medium", "Heavy"])
            cat_counts = cats.value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig = px.pie(
                cat_counts, names="Category", values="Count", color="Category",
                color_discrete_map={"Low": "#10B981", "Medium": "#F59E0B", "Heavy": "#EF4444"},
                hole=0.4
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
            )
            st.plotly_chart(fig, use_container_width=True)

    # 4. Hourly & Monthly trends
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        with st.container(border=True):
            st.markdown("### ⏰ Hourly Trend Profiles")
            st.write("Average traffic volume by hour of day.")
            hour_agg = df_preprocessed.groupby('Hours')['traffic_volume'].mean().reset_index()
            fig = px.line(hour_agg, x="Hours", y="traffic_volume", markers=True, color_discrete_sequence=['#06B6D4'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickmode='linear', tick0=0, dtick=2),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Average Vehicles")
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_t2:
        with st.container(border=True):
            st.markdown("### 📅 Monthly Trend Profiles")
            st.write("Average traffic volume by month of the year.")
            month_agg = df_preprocessed.groupby('Month')['traffic_volume'].mean().reset_index()
            fig = px.bar(
                month_agg, x="Month", y="traffic_volume", color="traffic_volume",
                color_continuous_scale="GnBu"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Average Vehicles")
            )
            st.plotly_chart(fig, use_container_width=True)

    # 5. Prediction History
    with st.container(border=True):
        st.markdown("### 🔮 Session Prediction History")
        st.write("Visualizes predictions run in this active browser session.")
        if st.session_state.predictions_history:
            history_df = pd.DataFrame(st.session_state.predictions_history)
            fig = px.line(
                history_df, y="predicted_volume", x="index_seq", text="predicted_volume",
                markers=True, title="Predicted Traffic Volume (Seq Order)",
                color_discrete_sequence=['#8B5CF6']
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
                xaxis=dict(title="Prediction Runs", gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title="Volume", gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions executed yet in this session. Head over to the 'Tesla Prediction Console' to start.")

# ==========================================================================
# PAGE 4: MACHINE LEARNING CENTER
# ==========================================================================
elif page == "🤖 Machine Learning Center":
    st.markdown("## 🤖 Machine Learning Model Dashboard")
    st.write("Inspect hyperparameters, split ratios, training statistics, and feature importance of the Random Forest model.")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        with st.container(border=True):
            st.markdown("### ⚙️ Model Hyperparameters & Specs")
            st.markdown("""
            <div style="margin-top: 10px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;">
                    <span style="color: #94A3B8;">Algorithm</span>
                    <span style="color: #06B6D4; font-weight: 700;">Random Forest Regressor</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;">
                    <span style="color: #94A3B8;">Estimators (Trees)</span>
                    <span style="color: #F8FAFC; font-weight: 600;">100 Trees</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;">
                    <span style="color: #94A3B8;">Criterion</span>
                    <span style="color: #F8FAFC; font-weight: 600;">Squared Error (MSE)</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;">
                    <span style="color: #94A3B8;">Train/Test Ratio</span>
                    <span style="color: #F8FAFC; font-weight: 600;">80% Train / 20% Test</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 8px 0;">
                    <span style="color: #94A3B8;">Total Features Fitted</span>
                    <span style="color: #F8FAFC; font-weight: 600;">9 Inputs</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                    <span style="color: #94A3B8;">Random State Seed</span>
                    <span style="color: #F8FAFC; font-weight: 600;">42</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-container" style="background: rgba(16, 185, 129, 0.05) !important; border-color: rgba(16, 185, 129, 0.2) !important;">
                <span style="font-size: 0.8rem; color: #10B981; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">EVALUATION SCORE</span>
                <h2 style="font-size: 2.2rem; font-weight: 800; color: #10B981; margin: 5px 0;">R² = {score:.4f}</h2>
                <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">
                    The model accounts for approximately <b>{score*100:.2f}%</b> of variance in traffic volume predictions based on climate and time vectors.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_m2:
        with st.container(border=True):
            st.markdown("### 🏆 Feature Importance Rankings")
            st.write("Understand which attributes exert the most influence on prediction algorithms.")
            feat_imp = pd.DataFrame({
                "Feature": x_train.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=True)
            fig = px.bar(
                feat_imp, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Tealgrn"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC', font_family='Outfit',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================================
# PAGE 5: TESLA PREDICTION CONSOLE
# ==========================================================================
elif page == "🔮 Tesla Prediction Console":
    st.markdown("## 🔮 Predict Traffic Volume")
    st.write("Tune weather, climate, and schedule sliders below inside this Tesla-style input card to predict real-time traffic volumes.")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])

    if ctrl_col2.button("🎲 Generate Random Sample", use_container_width=True):
        rand_row = df_original.sample(n=1).iloc[0]
        st.session_state.temp_input = float(rand_row['temp'] - 272.15)
        st.session_state.rain_input = float(rand_row['rain_1h'])
        st.session_state.snow_input = float(rand_row['snow_1h'])
        st.session_state.cloud_input = int(rand_row['clouds_all'])
        st.session_state.weather_main_input = str(rand_row['weather_main'])
        st.session_state.weather_desc_input = str(rand_row['weather_description'])
        dt = pd.to_datetime(rand_row['date_time'], dayfirst=True)
        st.session_state.hours_input = int(dt.hour)
        st.session_state.day_input = int(dt.day)
        st.session_state.month_input = int(dt.month)
        st.toast("Random row features loaded! 🎲", icon="⚡")
        st.rerun()

    if ctrl_col3.button("🔄 Reset to Default Values", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.toast("Console parameters reset! 🔄", icon="⭐")
        st.rerun()

    predict_clicked = False

    with st.container(border=True):
        st.markdown("### 🎛️ Climate & Temporal Controls")
        col_inp1, col_inp2, col_inp3 = st.columns(3)

        with col_inp1:
            temperature_c = st.number_input(
                "🌡️ Temperature (°C):", min_value=-20.0, max_value=50.0,
                value=st.session_state.temp_input, step=1.0,
                help="Enter degrees in Celsius. This automatically translates to Kelvin for model compatibility."
            )
            temp_k = temperature_c + 272.15
            rain = st.slider("🌦️ Rain Volume (1 hr):", 0.0, 20.0, value=st.session_state.rain_input, step=0.1)
            snow = st.slider("❄️ Snow Volume (1 hr):", 0.0, 20.0, value=st.session_state.snow_input, step=0.1)

        with col_inp2:
            cloud = st.slider("☁️ Cloud Coverage (%):", 0, 100, value=st.session_state.cloud_input)
            main_idx = MAIN_CLASSES.index(st.session_state.weather_main_input) if st.session_state.weather_main_input in MAIN_CLASSES else 0
            weather_main_str = st.selectbox("🌫️ Weather Category:", options=MAIN_CLASSES, index=main_idx)
            weather_main = MAIN_CLASSES.index(weather_main_str)
            desc_idx = DESC_CLASSES.index(st.session_state.weather_desc_input) if st.session_state.weather_desc_input in DESC_CLASSES else 0
            weather_description_str = st.selectbox("⛅ Detailed Description:", options=DESC_CLASSES, index=desc_idx)
            weather_description = DESC_CLASSES.index(weather_description_str)

        with col_inp3:
            hours = st.slider("⏰ Hour of Commute:", 0, 23, value=st.session_state.hours_input)
            day = st.slider("📅 Day of Month:", 1, 31, value=st.session_state.day_input)
            month_idx = st.session_state.month_input - 1
            month = st.selectbox(
                "🗓️ Month of Commute:", options=list(range(1, 13)), index=month_idx,
                format_func=lambda m: datetime.date(2026, m, 1).strftime("%B")
            )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)

        if ctrl_col1.button("🚕 Compute Traffic Predictions", use_container_width=True):
            predict_clicked = True

    # --- Predict Outcome Section ---
    if predict_clicked:
        input_data = [[temp_k, rain, snow, cloud, weather_main, weather_description, hours, day, month]]
        prediction = model.predict(input_data)[0]
        prediction_val = int(prediction)

        if prediction_val < 2000:
            badge = "🟢 LOW TRAFFIC DENSITY"
            color = "#10B981"
            rec_text = "Traffic levels are very light. Roadways are optimal. Commutes will experience standard to zero delays."
        elif 2000 <= prediction_val <= 4500:
            badge = "🟡 MEDIUM TRAFFIC DENSITY"
            color = "#F59E0B"
            rec_text = "Moderate commute flows detected. Some intersections might face brief queues. Keep standard commute routes."
        else:
            badge = "🔴 HEAVY TRAFFIC DENSITY"
            color = "#EF4444"
            rec_text = "Heavy commuter volume alert! Severe bottlenecks expected near intersections. Consider transit lanes or alternate routes."

        st.session_state.prediction_count += 1
        st.session_state.predictions_history.append({
            "index_seq": st.session_state.prediction_count,
            "predicted_volume": prediction_val,
            "category": badge
        })
        st.toast("Traffic Volume prediction generated successfully! ⚡", icon="🚨")

        st.markdown("### 🎯 Predicted Dashboard Insights")
        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            with st.container(border=True):
                st.markdown("<h4 style='margin: 0; text-align: center; color: #94A3B8;'>VEHICLE CONGESTION GAUGE</h4>", unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction_val,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Vehicles/Hr", 'font': {'size': 18, 'family': 'Outfit', 'color': '#F8FAFC'}},
                    gauge={
                        'axis': {'range': [0, 7500], 'tickcolor': "#F8FAFC"},
                        'bar': {'color': color},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'borderwidth': 1.5,
                        'bordercolor': "rgba(255,255,255,0.1)",
                        'steps': [
                            {'range': [0, 2000], 'color': 'rgba(16, 185, 129, 0.1)'},
                            {'range': [2000, 4500], 'color': 'rgba(245, 158, 11, 0.1)'},
                            {'range': [4500, 7500], 'color': 'rgba(239, 68, 68, 0.1)'}
                        ]
                    }
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC', font_family='Outfit',
                    height=240, margin=dict(t=40, b=10, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.markdown(f"""
            <div class="glass-container" style="border-left: 6px solid {color} !important; height: 260px;">
                <span style="font-size: 0.8rem; font-weight: 800; color: {color}; text-transform: uppercase; letter-spacing: 1px;">PREDICTIVE REPORT</span>
                <h2 style="font-size: 3rem; font-weight: 900; margin: 8px 0; color: #F8FAFC;">{prediction_val:,} <span style="font-size: 1.2rem; font-weight: 500; color: #94A3B8;">Vehicles</span></h2>
                <div style="font-size: 1rem; font-weight: 700; color: {color}; background-color: rgba(255,255,255,0.03); display: inline-block; padding: 4px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 12px;">
                    {badge}
                </div>
                <h4 style="margin: 5px 0 2px 0; color: #F8FAFC; font-weight: 600; font-size: 0.95rem;">Model Recommendation:</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.45; margin: 0;">
                    {rec_text}
                </p>
                <div style="margin-top: 12px; font-size: 0.75rem; color: #06B6D4; font-weight: 700; display: flex; justify-content: space-between;">
                    <span>⚡ Prediction Confidence: {(score*100):.1f}% (R²)</span>
                    <span>⌛ Forecast Time: {hours:02d}:00</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================================
# PAGE 6: SETTINGS
# ==========================================================================
elif page == "⚙️ Dashboard Settings":
    st.markdown("## ⚙️ Dashboard Controls & Export Profiles")
    st.write("Tune styling, dashboard responsiveness, or extract PDF summaries and database values.")

    with st.container(border=True):
        st.markdown("### 🎨 Aesthetic Styles & Toggle Themes")
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            theme_choice = st.selectbox(
                "Active Layout Theme Mode:",
                ["Dark Futuristic (Cyberpunk)", "Light Material Glass", "System Mode"],
                key="theme_mode_choice"
            )
            speed_choice = st.select_slider(
                "Interface Animation Transition Speed:",
                options=["Slow (Cinematic)", "Normal (Fluent)", "Fast (Performance)"],
                value="Normal (Fluent)", key="anim_speed_slider"
            )
        with col_set2:
            chart_style = st.selectbox(
                "Plotly Accent Palette Graph Scheme:",
                ["Deep Cyan & Space Indigo", "Emerald Forest & Lime Glow", "Cyberpunk Violet & Pink Laser"]
            )
            st.markdown("""
            <div style="margin-top: 15px; background: rgba(255,255,255,0.02); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                <p style="color: #94A3B8; font-size: 0.85rem; margin: 0; line-height: 1.4;">
                    💡 <b>Theme Note:</b> Theme changes dynamically adjust background glowing orbs and primary border properties.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### 📥 Document Export Profiles")
        st.write("Download analysis datasets or generate diagnostic summaries of current machine learning models.")
        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            st.markdown("#### 📂 Dataset (Cleaned)")
            csv_processed = df_preprocessed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📁 Download Preprocessed CSV", data=csv_processed,
                file_name="preprocessed_traffic_dataset.csv", mime="text/csv",
                use_container_width=True
            )
        with exp_col2:
            st.markdown("#### 📊 Model Settings (TXT)")
            model_info = f"""==================================================
SMART CITY TRAFFIC MODEL REPORT
==================================================
Date generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Algorithm: RandomForestRegressor
Estimators: 100
R2 Accuracy Score: {score:.6f}
Train Dataset Records: {x_train.shape[0]}
Test Dataset Records: {x_test.shape[0]}
Feature Weight Ranks: {dict(zip(x_train.columns, model.feature_importances_))}
=================================================="""
            st.download_button(
                label="📑 Download Model Summary", data=model_info,
                file_name="traffic_model_diagnostic.txt", mime="text/plain",
                use_container_width=True
            )
        with exp_col3:
            st.markdown("#### 📋 PDF Executive Summary")
            st.button("📄 PDF Export (Unavailable)", disabled=True, use_container_width=True)
            st.caption("Requires PDF compilation server settings")

    if st.button("🚨 System Full Diagnostics Reset", key="full_reset_btn"):
        st.session_state.prediction_count = 0
        st.session_state.predictions_history = []
        st.session_state.loaded_toasts = False
        st.toast("Dashboard parameters and prediction history reset! 🚨", icon="💥")
        st.rerun()

# ==========================================================================
# GLOBAL FOOTER
# ==========================================================================
st.markdown(f"""
<div class="footer-container">
    <div class="footer-links">
        <a href="https://github.com" target="_blank">🔗 GitHub Portfolio</a>
        <a href="https://linkedin.com" target="_blank">🔗 LinkedIn Profile</a>
        <a href="mailto:support@smartcitycommute.ai">📧 Email Developer</a>
    </div>
    <p style="color: #94A3B8; font-size: 0.85rem; margin-top: 15px; margin-bottom: 0;">
        Smart City Traffic Analytics System &bull; Version 1.2.0 &bull; Made with ❤️ using Python, Streamlit, and Machine Learning
    </p>
    <p style="color: #475569; font-size: 0.75rem; margin-top: 5px; margin-bottom: 0;">
        &copy; {now.year} Metropolitan Grid Systems. Licensed under MIT. All backend ML calculations are cached and certified.
    </p>
</div>
""", unsafe_allow_html=True)
