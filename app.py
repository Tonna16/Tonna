
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

# Set the page configuration for a wide layout
st.set_page_config(page_title="Ultimate Energy Monitoring System", layout="wide")

# Sidebar navigation for different sections
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Real-Time Weather", "Manual Input", "Prediction", "Data Export"])

if page == "Home":
    st.title("Ultimate Energy Monitoring System")
    st.write("Welcome to the advanced interactive energy monitoring and prediction app!")
    
    # Simulated energy data for the Home page (24 hours)
    time = np.arange(0, 24, 1)
    energy_usage = np.random.randint(100, 500, size=24)
    
    st.subheader("Energy Consumption Overview (Matplotlib)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, energy_usage, marker="o", linestyle="-", color="b")
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Energy Consumption (kWh)")
    ax.set_title("Energy Consumption Over 24 Hours")
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Interactive Energy Graph (Plotly)")
    df_home = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    fig_plotly = px.line(df_home, x="Time (Hours)", y="Energy Consumption (kWh)",
                         title="Interactive Energy Consumption", markers=True)
    st.plotly_chart(fig_plotly)

elif page == "Real-Time Weather":
    st.title("Real-Time Weather Data Integration")
    st.write("This section fetches real-time weather data that could be used to correlate with energy consumption trends.")
    
    # Input fields for weather API details
    API_KEY = st.text_input("Enter your OpenWeatherMap API Key", type="password")
    CITY = st.text_input("Enter City", value="Miami")
    
    if API_KEY and CITY:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data["main"]["temp"]
            st.write(f"Current Temperature in {CITY}: {temp} °C")
            # Display weather icon if available
            icon_code = data["weather"][0]["icon"]
            st.image(f"http://openweathermap.org/img/wn/{icon_code}@2x.png", width=100)
        else:
            st.error("Failed to retrieve weather data. Please check your API key or city name.")

elif page == "Manual Input":
    st.title("Manual Energy Data Input")
    st.write("Enter your energy consumption data (comma-separated) to visualize and analyze your own data.")
    
    # Text area for manual data input
    user_data = st.text_area("Enter energy consumption data (comma-separated):", "100, 200, 300, 400, 350, 275, 450, 300")
    try:
        # Process the input and generate a DataFrame
        user_energy = [float(x.strip()) for x in user_data.split(",") if x.strip() != ""]
        user_time = np.arange(0, len(user_energy))
        df_user = pd.DataFrame({"Time (Hours)": user_time, "Energy Consumption (kWh)": user_energy})
        st.subheader("Your Energy Consumption Data")
        st.dataframe(df_user)
        # Create an interactive Plotly graph for the user data
        fig_user = px.line(df_user, x="Time (Hours)", y="Energy Consumption (kWh)", 
                           title="User Input Energy Data", markers=True)
        st.plotly_chart(fig_user)
    except Exception as e:
        st.error("Invalid input. Please enter numbers separated by commas.")

elif page == "Prediction":
    st.title("Energy Consumption Prediction")
    st.write("Predict future energy consumption using a simple linear regression model.")
    
    # Use simulated data for model training (24 hours)
    time = np.arange(0, 24, 1)
    energy_usage = np.random.randint(100, 500, size=24)
    df_model = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    
    # Train the Linear Regression model
    X = df_model[["Time (Hours)"]]
    y = df_model["Energy Consumption (kWh)"]
    model = LinearRegression()
    model.fit(X, y)
    
    # Allow the user to choose a future hour for prediction (from 24 to 48)
    future_hour = st.slider("Select Future Hour for Prediction", 24, 48, 30)
    predicted_usage = model.predict([[future_hour]])
    st.write(f"Predicted Energy Consumption at Hour {future_hour}: {predicted_usage[0]:.2f} kWh")
    
    # Display prediction graph over the future time range
    future_time = np.arange(24, 49, 1)
    future_predictions = model.predict(future_time.reshape(-1, 1))
    df_pred = pd.DataFrame({"Time (Hours)": future_time, "Predicted Energy Consumption (kWh)": future_predictions})
    fig_pred = px.line(df_pred, x="Time (Hours)", y="Predicted Energy Consumption (kWh)", 
                       title="Predicted Future Energy Consumption", markers=True)
    st.plotly_chart(fig_pred)

elif page == "Data Export":
    st.title("Download Energy Data")
    st.write("Download your current energy consumption data as a CSV file.")
    
    # For demonstration, we use simulated data for export
    time = np.arange(0, 24, 1)
    energy_usage = np.random.randint(100, 500, size=24)
    df_export = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "energy_data.csv", "text/csv"