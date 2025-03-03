import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 60 seconds (60000 ms)
st_autorefresh(interval=60000, limit=100, key="energy_autorefresh")

# Initialize session state for settings if not already set
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"

# Language translations (for demonstration)
translations = {
    "English": {
        "title": "Ultimate Energy Monitoring System",
        "welcome": "Welcome to the advanced interactive energy monitoring and prediction app!"
    },
    "Spanish": {
        "title": "Sistema de Monitoreo de Energía Definitivo",
        "welcome": "¡Bienvenido a la aplicación avanzada de monitoreo y predicción de energía!"
    }
}

# Apply custom CSS based on selected theme
if st.session_state.theme == "Dark":
    custom_css = """
    <style>
    body { background-color: #2E2E2E; color: #FFFFFF; }
    .sidebar-content { background-color: #424242; }
    </style>
    """
elif st.session_state.theme == "Blue":
    custom_css = """
    <style>
    body { background-color: #e0f7fa; color: #006064; }
    .sidebar-content { background-color: #b2ebf2; }
    </style>
    """
else:  # Light theme
    custom_css = """
    <style>
    body { background-color: #FFFFFF; color: #000000; }
    .sidebar-content { background-color: #f0f0f0; }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar navigation including the new Settings page
st.sidebar.title("Navigation")
pages = [
    "Home", 
    "Custom Tracker", 
    "Real-Time Weather", 
    "Manual Input", 
    "Prediction", 
    "Advanced Analysis", 
    "Premium Recommendations", 
    "Gamification & Optimization", 
    "Data Export",
    "Settings"
]
page = st.sidebar.radio("Go to", pages)

# ----------------------------
# Helper function to simulate energy data
def get_simulated_energy_data(hours=24):
    time = np.arange(0, hours, 1)
    energy_usage = np.random.randint(100, 500, size=hours)
    return time, energy_usage

# ----------------------------
# SETTINGS PAGE: Let users choose language and theme
if page == "Settings":
    st.title("Settings")
    st.write("Customize the app settings to your liking.")

    # Language selection
    lang = st.selectbox("Select Language", ["English", "Spanish"], index=["English", "Spanish"].index(st.session_state.language))
    st.session_state.language = lang

    # Theme selection
    theme = st.selectbox("Select Theme", ["Light", "Dark", "Blue"], index=["Light", "Dark", "Blue"].index(st.session_state.theme))
    st.session_state.theme = theme

    st.write("Settings saved! Your app will now use the selected language and theme.")
    st.write("Current Language:", st.session_state.language)
    st.write("Current Theme:", st.session_state.theme)

# ----------------------------
# HOME PAGE: Overview of Energy Consumption
elif page == "Home":
    st.title(translations[st.session_state.language]["title"])
    st.write(translations[st.session_state.language]["welcome"])
    
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

# ----------------------------
# CUSTOM TRACKER: Appliance-based energy tracking
elif page == "Custom Tracker":
    st.title("Custom Energy Consumption Tracker")
    st.write("Select appliances, input usage hours, and calculate your daily energy consumption and cost.")
    
    appliance_data = {
        "Air Conditioner": 2000,
        "Refrigerator": 150,
        "Washing Machine": 500,
        "TV": 100,
        "Laptop": 50,
        "Microwave": 1200,
        "Lights": 60,
        "Fan": 75
    }
    
    selected_appliances = st.multiselect("Select Appliances", list(appliance_data.keys()))
    appliance_usage = {}
    for appliance in selected_appliances:
        hours = st.number_input(f"Hours used per day ({appliance})", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        appliance_usage[appliance] = hours
    
    cost_rate = st.number_input("Enter cost per kWh ($)", min_value=0.0, value=0.12, step=0.01)
    
    total_energy = 0
    breakdown = []
    for appliance, hours in appliance_usage.items():
        power_kw = appliance_data[appliance] / 1000
        energy = power_kw * hours
        total_energy += energy
        breakdown.append((appliance, energy))
    
    total_cost = total_energy * cost_rate
    st.write(f"**Total Energy Consumption:** {total_energy:.2f} kWh per day")
    st.write(f"**Total Cost:** ${total_cost:.2f} per day")
    
    df_breakdown = pd.DataFrame(breakdown, columns=["Appliance", "Energy (kWh)"])
    st.table(df_breakdown)
    fig_breakdown = px.bar(df_breakdown, x="Appliance", y="Energy (kWh)", title="Energy Consumption per Appliance")
    st.plotly_chart(fig_breakdown)

# ----------------------------
# REAL-TIME WEATHER: Fetch live weather data based on city input
elif page == "Real-Time Weather":
    st.title("Real-Time Weather Data Integration")
    st.write("Enter a city name to fetch live weather data.")
    
    default_api_key = "500c92f90ed8f4f84755b89b9e05e714"  # Replace with your API key
    city_input = st.text_input("Enter a city name", "New York")
    
    def get_weather(city, api_key=default_api_key):
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                "City": data["name"],
                "Temperature (°C)": data["main"]["temp"],
                "Weather": data["weather"][0]["description"].capitalize(),
                "Humidity (%)": data["main"]["humidity"],
                "Wind Speed (m/s)": data["wind"]["speed"]
            }
            return weather_info
        else:
            return {"Error": "City not found. Please try again."}
    
    if st.button("Get Weather"):
        weather_data = get_weather(city_input)
        if "Error" in weather_data:
            st.error(weather_data["Error"])
        else:
            st.success(f"Weather in {weather_data['City']}: {weather_data['Weather']}")
            st.write(f"🌡️ Temperature: {weather_data['Temperature (°C)']}°C")
            st.write(f"💧 Humidity: {weather_data['Humidity (%)']}%")
            st.write(f"💨 Wind Speed: {weather_data['Wind Speed (m/s)']} m/s")

# ----------------------------
# MANUAL INPUT: User-entered energy data visualization
elif page == "Manual Input":
    st.title("Manual Energy Data Input")
    st.write("Enter your own energy consumption data (comma-separated) to visualize and analyze.")
    
    user_data = st.text_area("Enter energy consumption data (comma-separated):", "100,200,300,400,350,275,450,300")
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

# ----------------------------
# PREDICTION: AI-powered energy consumption prediction with advanced features
elif page == "Prediction":
    st.title("Energy Consumption Prediction")
    st.write("Predict future energy consumption using advanced feature engineering and machine learning.")
    
    hours = np.arange(0, 24)
    base = 150
    amplitude = 200
    noise = np.random.normal(0, 20, size=24)
    energy_usage = base + amplitude * np.sin((hours - 6) * np.pi / 12) + noise
    
    df_model = pd.DataFrame({
        "Hour": hours,
        "sin_hour": np.sin(hours * 2 * np.pi / 24),
        "cos_hour": np.cos(hours * 2 * np.pi / 24),
        "Energy Consumption": energy_usage
    })
    
    X = df_model[["Hour", "sin_hour", "cos_hour"]]
    y = df_model["Energy Consumption"]
    
    model_type = st.selectbox("Select Prediction Model", ["Linear Regression", "Random Forest"])
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        model.fit(X, y)
    
    future_hour = st.slider("Select Future Hour for Prediction", 24, 48, 30)
    future_features = np.array([
        future_hour,
        np.sin(future_hour * 2 * np.pi / 24),
        np.cos(future_hour * 2 * np.pi / 24)
    ]).reshape(1, -1)
    
    predicted_usage = model.predict(future_features)
    st.write(f"Predicted Energy Consumption at Hour {future_hour}: {predicted_usage[0]:.2f} kWh")
    
    future_hours = np.arange(24, 49)
    future_features_all = np.column_stack((
        future_hours,
        np.sin(future_hours * 2 * np.pi / 24),
        np.cos(future_hours * 2 * np.pi / 24)
    ))
    future_predictions = model.predict(future_features_all)
    df_pred = pd.DataFrame({"Hour": future_hours, "Predicted Energy Consumption (kWh)": future_predictions})
    fig_pred = px.line(df_pred, x="Hour", y="Predicted Energy Consumption (kWh)",
                       title=f"{model_type} - Predicted Future Energy Consumption", markers=True)
    st.plotly_chart(fig_pred)

# ----------------------------
# ADVANCED ANALYSIS: Detailed analytics with statistics and trend analysis
elif page == "Advanced Analysis":
    st.title("Advanced Energy Analysis")
    st.write("Explore detailed analytics on energy data: statistical summaries, correlations, and trend analysis.")
    
    time, energy_usage = get_simulated_energy_data(48)
    df_adv = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    
    st.subheader("Statistical Overview")
    st.write(df_adv.describe())
    
    st.subheader("Correlation Matrix")
    corr = df_adv.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr)
    
    st.subheader("Scatter Plot Analysis with Trendline")
    fig_scatter = px.scatter(df_adv, x="Time (Hours)", y="Energy Consumption (kWh)",
                             title="Energy Consumption Scatter Plot with Trendline", trendline="ols")
    st.plotly_chart(fig_scatter)

# ----------------------------
# PREMIUM RECOMMENDATIONS: Insights and cost-saving suggestions
elif page == "Premium Recommendations":
    st.title("Premium Recommendations")
    st.write("Unlock insights to reduce energy consumption and save money!")
    
    time, energy_usage = get_simulated_energy_data(24)
    total_energy = np.sum(energy_usage)
    avg_consumption = np.mean(energy_usage)
    st.write(f"**Total Energy Consumption Today:** {total_energy:.2f} kWh")
    st.write(f"**Average Hourly Consumption:** {avg_consumption:.2f} kWh")
    
    if avg_consumption > 300:
        st.error("Your energy usage is above average. Consider turning off non-essential appliances during peak hours!")
    else:
        st.success("Great job! Your energy usage is within a good range. Keep it up!")
    
    cost_rate = st.number_input("Enter cost per kWh ($)", min_value=0.0, value=0.12, step=0.01)
    daily_cost = total_energy * cost_rate
    st.write(f"**Estimated Daily Cost:** ${daily_cost:.2f}")
    
    monthly_reduction = st.slider("What % reduction in usage can you achieve?", 0, 50, 10)
    potential_savings = daily_cost * (monthly_reduction/100) * 30
    st.write(f"By reducing your consumption by {monthly_reduction}%, you could save approximately **${potential_savings:.2f} per month**!")
    
    fig_premium = px.line(x=time, y=energy_usage, labels={"x": "Time (Hours)", "y": "Energy Consumption (kWh)"},
                          title="Your Daily Energy Consumption Trend")
    st.plotly_chart(fig_premium)

# ----------------------------
# GAMIFICATION & OPTIMIZATION: Daily challenges, streaks, and personalized suggestions
elif page == "Gamification & Optimization":
    st.title("Gamification & Optimization")
    st.write("Engage in challenges, track your streaks, and receive personalized energy-saving suggestions!")
    
    st.subheader("Daily Energy Savings Challenge")
    target_reduction = st.slider("Select your target energy reduction (%) for today:", 0, 50, 10)
    
    _, simulated_consumption = get_simulated_energy_data(24)
    total_consumption = np.sum(simulated_consumption)
    st.write(f"Simulated total energy consumption today: {total_consumption:.2f} kWh")
    
    baseline_consumption = 600
    reduction_amount = baseline_consumption * (target_reduction / 100)
    target_consumption = baseline_consumption - reduction_amount
    st.write(f"Your target consumption for today: {target_consumption:.2f} kWh (Baseline: {baseline_consumption} kWh)")
    
    if total_consumption <= target_consumption:
        st.success("Congratulations! You met today's energy savings challenge!")
        if 'streak' not in st.session_state:
            st.session_state['streak'] = 1
        else:
            st.session_state['streak'] += 1
    else:
        st.error("You did not meet the challenge today. Try to improve tomorrow!")
        if 'streak' not in st.session_state:
            st.session_state['streak'] = 0
    st.write(f"**Current Challenge Streak:** {st.session_state.get('streak', 0)} days")
    
    st.subheader("Community Leaderboard")
    leaderboard = pd.DataFrame({
        "User": ["Alice", "Bob", "Charlie", "You", "Diana"],
        "Streak (days)": [5, 3, 7, st.session_state.get('streak', 0), 4],
        "Savings Achieved (%)": [15, 10, 20, target_reduction, 12]
    })
    st.table(leaderboard.sort_values(by="Streak (days)", ascending=False))
    
    st.subheader("Personalized Energy-Saving Suggestions")
    avg_consumption = total_consumption / 24
    suggestions = []
    if avg_consumption > 30:
        suggestions.append("Consider reducing HVAC usage during peak hours.")
    if total_consumption > baseline_consumption:
        suggestions.append("Turn off non-essential appliances when not in use.")
    else:
        suggestions.append("Great job! Keep maintaining your energy-saving habits.")
    st.write("**Suggestions:**")
    for sug in suggestions:
        st.write(f"- {sug}")

# ----------------------------
# DATA EXPORT: Download simulated energy data as CSV
elif page == "Data Export":
    st.title("Download Energy Data")
    st.write("Download the simulated energy consumption data as a CSV file.")
    time, energy_usage = get_simulated_energy_data(24)
    df_export = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "energy_data.csv", "text/csv")





