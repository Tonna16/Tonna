import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Configure the page layout and title
st.set_page_config(page_title="Ultimate Energy Monitoring System", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Real-Time Weather", "Manual Input", "Prediction", "Advanced Analysis", "Data Export"]
page = st.sidebar.radio("Go to", pages)

# Helper function for simulated energy data
def get_simulated_energy_data(hours=24):
    time = np.arange(0, hours, 1)
    energy_usage = np.random.randint(100, 500, size=hours)
    return time, energy_usage

if page == "Home":
    st.title("Ultimate Energy Monitoring System")
    st.write("Welcome to the advanced interactive energy monitoring and prediction app!")
    
    # Home Page: Display energy graphs using simulated data (24 hours)
    time, energy_usage = get_simulated_energy_data(24)
    
    st.subheader("Energy Consumption Overview (Matplotlib)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, energy_usage, marker="o", linestyle="-", color="b", label="Energy Usage")
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.set_title("Energy Consumption Over 24 Hours")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Interactive Energy Graph (Plotly)")
    df_home = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    fig_plotly = px.line(df_home, x="Time (Hours)", y="Energy Consumption (kWh)",
                         title="Interactive Energy Consumption", markers=True)
    st.plotly_chart(fig_plotly)

elif page == "Real-Time Weather":
    st.title("Real-Time Weather Data Integration")
    st.write("This section fetches live weather data to correlate with energy trends.")
    
    # Input fields for weather API details (optional)
    API_KEY = st.text_input("Enter your OpenWeatherMap API Key (optional)", type="password")
    CITY = st.text_input("Enter City", value="Miami")
    
    if API_KEY:
        @st.cache_data(ttl=600)
        def get_weather(api_key, city):
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        data = get_weather(API_KEY, CITY)
        if data:
            temp = data["main"]["temp"]
            st.write(f"Current Temperature in {CITY}: {temp} °C")
            icon_code = data["weather"][0]["icon"]
            st.image(f"http://openweathermap.org/img/wn/{icon_code}@2x.png", width=100)
        else:
            st.error("Failed to retrieve weather data. Please check your API key and city name.")
    else:
        st.info("No API key provided. Weather data is unavailable.")

elif page == "Manual Input":
    st.title("Manual Energy Data Input")
    st.write("Enter your energy consumption data (comma-separated) to visualize and analyze your own data.")
    
    user_data = st.text_area("Enter energy consumption data (comma-separated):", "100, 200, 300, 400, 350, 275, 450, 300")
    try:
        user_energy = [float(x.strip()) for x in user_data.split(",") if x.strip() != ""]
        user_time = np.arange(0, len(user_energy))
        df_user = pd.DataFrame({"Time (Hours)": user_time, "Energy Consumption (kWh)": user_energy})
        st.subheader("Your Energy Consumption Data")
        st.dataframe(df_user)
        fig_user = px.line(df_user, x="Time (Hours)", y="Energy Consumption (kWh)", 
                           title="User Input Energy Data", markers=True)
        st.plotly_chart(fig_user)
    except Exception as e:
        st.error("Invalid input. Please enter numbers separated by commas.")

elif page == "Prediction":
    st.title("Energy Consumption Prediction")
    st.write("Predict future energy consumption using a machine learning model. Customize your prediction model below.")
    
    # Let users choose a prediction model
    model_type = st.selectbox("Select Prediction Model", ["Linear Regression", "Random Forest"])
    
    # Use simulated data for model training (24 hours)
    time, energy_usage = get_simulated_energy_data(24)
    df_model = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    X = df_model[["Time (Hours)"]]
    y = df_model["Energy Consumption (kWh)"]
    
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
    
    future_hour = st.slider("Select Future Hour for Prediction", 24, 48, 30)
    predicted_usage = model.predict([[future_hour]])
    st.write(f"Predicted Energy Consumption at Hour {future_hour}: {predicted_usage[0]:.2f} kWh")
    
    # Plot predictions over a future time range
    future_time = np.arange(24, 49, 1)
    future_predictions = model.predict(future_time.reshape(-1, 1))
    df_pred = pd.DataFrame({"Time (Hours)": future_time, "Predicted Energy Consumption (kWh)": future_predictions})
    fig_pred = px.line(df_pred, x="Time (Hours)", y="Predicted Energy Consumption (kWh)", 
                       title=f"{model_type} - Predicted Future Energy Consumption", markers=True)
    st.plotly_chart(fig_pred)

elif page == "Advanced Analysis":
    st.title("Advanced Energy Analysis")
    st.write("Explore advanced analytics on energy data including statistical summaries, correlation, and trend analysis.")
    
    # Simulate a larger dataset (48 hours) for advanced analysis
    time, energy_usage = get_simulated_energy_data(48)
    df_adv = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    
    st.subheader("Statistical Overview")
    st.write(df_adv.describe())
    
    st.subheader("Correlation Matrix")
    corr = df_adv.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr)
    
    st.subheader("Scatter Plot Analysis")
    fig_scatter = px.scatter(df_adv, x="Time (Hours)", y="Energy Consumption (kWh)", 
                             title="Energy Consumption Scatter Plot with Trendline", trendline="ols")
    st.plotly_chart(fig_scatter)

elif page == "Data Export":
    st.title("Download Energy Data")
    st.write("Download the simulated energy consumption data as a CSV file.")
    time, energy_usage = get_simulated_energy_data(24)
    df_export = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "energy_data.csv", "text/csv")







