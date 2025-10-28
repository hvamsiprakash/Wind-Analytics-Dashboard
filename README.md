# 🌬️ Wind Energy Analytics Dashboard

  
🔗 **Live App:** [Wind Energy Analytics Dashboard](https://wind-analytics-dashboard-3dv5dfpquv6xawtmbnwf8q.streamlit.app/)

***

## 🚀 Features

- **Wind Data Visualization:** Time series, wind rose, polar plots, and density heatmaps.
- **Turbine Simulation:** Compare major turbine models or configure custom ones.
- **Energy Forecasting:** Predict power and cumulative energy from local weather.
- **Machine Learning:** Random Forest regression for robust wind speed predictions.
- **Model Validation:** MAE, RMSE, and R² metrics, plus actual vs predicted comparisons.
- **Interactive UI:** Sidebar configuration, filtering, live charts, and data downloads.

***



## 🧰 Tech Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Data:** Open-Meteo API, OpenStreetMap Nominatim
- **ML Model:** Scikit-learn Random Forest
- **Languages:** Python

***

## ⚡ Installation

```bash
git clone https://github.com/yourusername/wind-energy-dashboard.git
cd wind-energy-dashboard
python -m venv venv
source venv/bin/activate   # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt
streamlit run app.py
```

***

## 🧾 Requirements

```
streamlit
requests
pandas
numpy
plotly
scikit-learn
matplotlib
scipy
windrose
```

***

## 🔍 Usage

1. Select a location, turbine model, and analysis type from the sidebar.
2. Click "**Analyze Wind Data**" to fetch forecasts and generate dashboard metrics/plots.
3. View interactive analysis tabs:
   - Wind Analysis
   - Turbine Performance
   - Energy Forecast
   - Wind Prediction & Model Validation
4. Download raw CSV data for further analysis.

***

## 💡 How Does It Work?

- **Geocoding:** Converts city/country to latitude/longitude using OpenStreetMap.
- **Weather Forecasts:** Fetches hourly wind, temperature, humidity data from Open-Meteo.
- **Turbine Modeling:** Built-in library for classic wind turbine specifications.
- **Energy Estimation:** Calculates output based on wind speed and air density.
- **ML Prediction:** Uses Random Forest for 6–48 hour wind forecasts with confidence bands.
- **Validation:** Benchmarks model predictions against actual historic observations.

***

## 📚 Model Details

- **Algorithm:** RandomForestRegressor (200 trees, max depth 10)
- **Features:** Hour-of-day (sin/cos), weather variables, historical lags, calendar info
- **Evaluation:** MAE, RMSE, R² – visible in the Wind Prediction tab

***

## 🌍 Live Demo

Try it here:  
**[https://wind-analytics-dashboard-3dv5dfpquv6xawtmbnwf8q.streamlit.app/](https://wind-analytics-dashboard-3dv5dfpquv6xawtmbnwf8q.streamlit.app/)**

***


