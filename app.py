import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import requests
from voice_commands import voice_commands_page
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from streamlit_autorefresh import st_autorefresh
from fpdf import FPDF
# Uncomment if using deep learning later
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# ----------------------------
# Auto-refresh every 60 seconds (simulate real-time updates)
st_autorefresh(interval=60000, limit=100, key="energy_autorefresh")

# Set page configuration
st.set_page_config(page_title="Ultimate Energy Monitoring System", layout="wide")

# ----------------------------
# SESSION STATE DEFAULTS
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'temp_unit' not in st.session_state:
    st.session_state.temp_unit = "Celsius"
if 'streak' not in st.session_state:
    st.session_state['streak'] = 0
if 'manual_data' not in st.session_state:
    st.session_state['manual_data'] = None

# ----------------------------
# TRANSLATIONS: All UI text is managed here
translations = {
    "English": {
        "title": "Ultimate Energy Monitoring System",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "login_button": "Log In",
        "login_error": "Incorrect username or password.",
        "logout": "Log Out",
        "custom_tracker": "Custom Energy Consumption Tracker",
        "weather": "Real-Time Weather Data",
        "weather_input": "Enter a city name",
        "weather_button": "Get Weather",
        "weather_temp": "Temperature",
        "weather_humidity": "Humidity",
        "weather_wind": "Wind Speed",
        "prediction": "Energy Consumption Prediction",
        "prediction_model": "Select Prediction Model",
        "prediction_lstm": "LSTM (Coming Soon)",
        "use_manual_data": "Use my manual input data for prediction",
        "premium": "Premium Recommendations",
        "gamification": "Gamification & Optimization",
        "data_export": "Download Energy Data",
        "manual_input": "Manual Energy Data Input",
        "chat_assistant": "Chat Assistant",
        "iot_integration": "IoT Integration",
        "voice_commands": "Voice Commands (Coming Soon)",
        "settings": "Settings",
        "settings_language": "Select Language",
        "settings_theme": "Select Theme",
        "settings_temp_unit": "Select Temperature Unit",
        "settings_dashboard": "Select Dashboard Pages to Show",
        "challenge_success": "Congratulations! You met today's energy savings challenge!",
        "challenge_failure": "You did not meet the challenge today. Try to improve tomorrow!",
        "streak": "Current Challenge Streak",
        "leaderboard": "Community Leaderboard",
        "chat_prompt": "Ask for an energy-saving tip:",
        "chat_button": "Ask"
    },
    "Spanish": {
        "title": "Sistema de Monitoreo de Energía Definitivo",
        "login": "Iniciar Sesión",
        "username": "Nombre de Usuario",
        "password": "Contraseña",
        "login_button": "Acceder",
        "login_error": "Usuario o contraseña incorrectos.",
        "logout": "Cerrar Sesión",
        "custom_tracker": "Rastreador de Consumo de Energía Personalizado",
        "weather": "Datos Meteorológicos en Tiempo Real",
        "weather_input": "Ingrese el nombre de una ciudad",
        "weather_button": "Obtener Clima",
        "weather_temp": "Temperatura",
        "weather_humidity": "Humedad",
        "weather_wind": "Velocidad del Viento",
        "prediction": "Predicción del Consumo de Energía",
        "prediction_model": "Seleccione el modelo de predicción",
        "prediction_lstm": "LSTM (Próximamente)",
        "use_manual_data": "Usar mis datos de entrada manual para la predicción",
        "premium": "Recomendaciones Premium",
        "gamification": "Gamificación y Optimización",
        "data_export": "Descargar Datos de Energía",
        "manual_input": "Entrada Manual de Datos de Energía",
        "chat_assistant": "Asistente de Chat",
        "iot_integration": "Integración IoT",
        "voice_commands": "Comandos de Voz (Próximamente)",
        "settings": "Configuración",
        "settings_language": "Seleccione el idioma",
        "settings_theme": "Seleccione el tema",
        "settings_temp_unit": "Seleccione la unidad de temperatura",
        "settings_dashboard": "Seleccione las páginas del panel a mostrar",
        "challenge_success": "¡Felicidades! ¡Ha alcanzado el reto de ahorro de energía de hoy!",
        "challenge_failure": "No alcanzó el reto hoy. ¡Inténtelo de nuevo mañana!",
        "streak": "Racha actual de desafíos",
        "leaderboard": "Tabla de Clasificación",
        "chat_prompt": "Pida un consejo para ahorrar energía:",
        "chat_button": "Preguntar"
    }
}

def tr(key):
    return translations[st.session_state.language].get(key, key)

# ----------------------------
# CUSTOM CSS BASED ON THEME
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
else:
    custom_css = """
    <style>
    body { background-color: #FFFFFF; color: #000000; }
    .sidebar-content { background-color: #f0f0f0; }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------------------
# LOGIN SYSTEM: Simple demo (replace with secure auth in production)
def show_login():
    st.title(tr("login"))
    username = st.text_input(tr("username"))
    password = st.text_input(tr("password"), type="password")
    if st.button(tr("login_button")):
        # Demo credentials (in production, verify against a database)
        if username == "admin" and password == "password":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
        else:
            st.error(tr("login_error"))

# If not logged in, only allow access to Login and Settings pages.
if not st.session_state.logged_in:
    page = st.sidebar.radio("Go to", ["Login", "Settings"])
    if page == "Login":
        show_login()
    elif page == "Settings":
        # Show settings even if not logged in
        st.title(tr("settings"))
        lang = st.selectbox(tr("settings_language"), ["English", "Spanish"], index=["English", "Spanish"].index(st.session_state.language))
        st.session_state.language = lang
        theme = st.selectbox(tr("settings_theme"), ["Light", "Dark", "Blue"], index=["Light", "Dark", "Blue"].index(st.session_state.theme))
        st.session_state.theme = theme
        temp_unit = st.selectbox(tr("settings_temp_unit"), ["Celsius", "Fahrenheit"], index=["Celsius", "Fahrenheit"].index(st.session_state.temp_unit))
        st.session_state.temp_unit = temp_unit
        st.write("Settings updated!")
        st.write(f"{tr('settings_language')}: {st.session_state.language}")
        st.write(f"{tr('settings_theme')}: {st.session_state.theme}")
        st.write(f"{tr('settings_temp_unit')}: {st.session_state.temp_unit}")
    st.stop()  # Prevent access to other pages until login

# ----------------------------
# SIDEBAR NAVIGATION: Only if logged in
st.sidebar.title("Navigation")
all_pages = ["Custom Tracker", "Real-Time Weather", "Prediction", "Premium Recommendations", "Gamification & Optimization", "Data Export", "Manual Input", "Chat Assistant", "IoT Integration", "Voice Commands", "Settings", "Logout"]
selected_pages = st.sidebar.multiselect("Dashboard Pages", options=all_pages, default=all_pages)
st.session_state.dashboard_pages = selected_pages
page = st.sidebar.radio("Go to", st.session_state.dashboard_pages)

# Logout function
if page == "Logout":
    st.session_state.logged_in = False
    st.experimental_rerun()

# ----------------------------
# CUSTOM TRACKER PAGE
if page == "Custom Tracker":
    st.title(tr("custom_tracker"))
    st.write("Select your appliances, enter daily usage, and calculate your energy consumption and cost.")
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
    st.write(f"**Total Energy Consumption:** {total_energy:.2f} kWh/day")
    st.write(f"**Total Cost:** ${total_cost:.2f}/day")
    df_breakdown = pd.DataFrame(breakdown, columns=["Appliance", "Energy (kWh)"])
    st.table(df_breakdown)
    fig_breakdown = px.bar(df_breakdown, x="Appliance", y="Energy (kWh)", title="Energy Consumption per Appliance")
    st.plotly_chart(fig_breakdown)

# ----------------------------
# REAL-TIME WEATHER PAGE
elif page == "Real-Time Weather":
    st.title(tr("weather"))
    st.write("Enter a city name to get live weather data.")
    default_api_key = "YOUR_OPENWEATHER_API_KEY"  # Replace with your API key
    city_input = st.text_input(tr("weather_input"), "New York")
    def get_weather(city, api_key=default_api_key):
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            temp_c = data["main"]["temp"]
            if st.session_state.temp_unit == "Fahrenheit":
                temp = temp_c * 9/5 + 32
                unit = "°F"
            else:
                temp = temp_c
                unit = "°C"
            weather_info = {
                "City": data["name"],
                "Temperature": f"{temp:.1f} {unit}",
                "Weather": data["weather"][0]["description"].capitalize(),
                "Humidity": f"{data['main']['humidity']}%",
                "Wind Speed": f"{data['wind']['speed']} m/s"
            }
            if (st.session_state.temp_unit == "Celsius" and temp > 35) or ("storm" in data["weather"][0]["description"].lower()):
                st.warning("Severe weather alert: High temperature or storm conditions detected!")
            return weather_info
        else:
            return {"Error": "City not found. Please try again."}
    if st.button(tr("weather_button")):
        weather_data = get_weather(city_input)
        if "Error" in weather_data:
            st.error(weather_data["Error"])
        else:
            st.success(f"Weather in {weather_data['City']}: {weather_data['Weather']}")
            st.write(f"{tr('weather_temp')}: {weather_data['Temperature']}")
            st.write(f"{tr('weather_humidity')}: {weather_data['Humidity']}")
            st.write(f"{tr('weather_wind')}: {weather_data['Wind Speed']}")

# ----------------------------
# PREDICTION PAGE
elif page == "Prediction":
    st.title(tr("prediction"))
    st.write("Predict future energy consumption using advanced machine learning.")
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
    model_choice = st.selectbox(tr("prediction_model"), ["Linear Regression", "Random Forest", tr("prediction_lstm")])
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        model.fit(X, y)
    else:
        st.info("LSTM forecasting is coming soon. Using Random Forest as default.")
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
    future_preds = model.predict(future_features_all)
    df_pred = pd.DataFrame({"Hour": future_hours, "Predicted Energy Consumption (kWh)": future_preds})
    fig_pred = px.line(df_pred, x="Hour", y="Predicted Energy Consumption (kWh)",
                       title=f"{model_choice} - Predicted Future Energy Consumption", markers=True)
    st.plotly_chart(fig_pred)

# ----------------------------
# PREMIUM RECOMMENDATIONS PAGE
elif page == "Premium Recommendations":
    st.title(tr("premium"))
    st.write("Unlock insights to reduce energy consumption and save money!")
    time, energy_usage = get_simulated_energy_data(24)
    total_energy = np.sum(energy_usage)
    avg_consumption = np.mean(energy_usage)
    st.write(f"**Total Energy Consumption Today:** {total_energy:.2f} kWh")
    st.write(f"**Average Hourly Consumption:** {avg_consumption:.2f} kWh")
    if avg_consumption > 300:
        st.error("Your energy usage is above average. Consider turning off non-essential appliances during peak hours!")
    else:
        st.success("Great job! Your energy usage is within a good range.")
    cost_rate = st.number_input("Enter cost per kWh ($)", min_value=0.0, value=0.12, step=0.01)
    daily_cost = total_energy * cost_rate
    st.write(f"**Estimated Daily Cost:** ${daily_cost:.2f}")
    monthly_reduction = st.slider("What % reduction in usage can you achieve?", 0, 50, 10)
    potential_savings = daily_cost * (monthly_reduction / 100) * 30
    st.write(f"By reducing your consumption by {monthly_reduction}%, you could save approximately **${potential_savings:.2f} per month**!")
    fig_premium = px.line(x=time, y=energy_usage, labels={"x": "Time (Hours)", "y": "Energy Consumption (kWh)"},
                          title="Your Daily Energy Consumption Trend")
    st.plotly_chart(fig_premium)

# ----------------------------
# GAMIFICATION & OPTIMIZATION PAGE
elif page == "Gamification & Optimization":
    st.title(tr("gamification"))
    st.write("Engage in daily challenges and receive personalized energy-saving suggestions!")
    st.subheader("Daily Energy Savings Challenge")
    target_reduction = st.slider("Select your target energy reduction (%) for today:", 0, 50, 10)
    _, simulated_consumption = get_simulated_energy_data(24)
    total_consumption = np.sum(simulated_consumption)
    st.write(f"Simulated total energy consumption today: {total_consumption:.2f} kWh")
    baseline = 600
    reduction_amount = baseline * (target_reduction / 100)
    target_consumption = baseline - reduction_amount
    st.write(f"Your target consumption for today: {target_consumption:.2f} kWh (Baseline: {baseline} kWh)")
    if total_consumption <= target_consumption:
        st.success(tr("challenge_success"))
        st.session_state['streak'] += 1
    else:
        st.error(tr("challenge_failure"))
        st.session_state['streak'] = 0
    st.write(f"**{tr('streak')}:** {st.session_state.get('streak', 0)} days")
    st.subheader(tr("leaderboard"))
    leaderboard = pd.DataFrame({
        "User": ["Alice", "Bob", "Charlie", "You", "Diana"],
        "Streak (days)": [5, 3, 7, st.session_state.get('streak', 0), 4],
        "Savings Achieved (%)": [15, 10, 20, target_reduction, 12]
    })
    st.table(leaderboard.sort_values(by="Streak (days)", ascending=False))
    st.subheader("Personalized Energy-Saving Suggestions")
    avg_cons = total_consumption / 24
    suggestions = []
    if avg_cons > 30:
        suggestions.append("Consider reducing HVAC usage during peak hours.")
    if total_consumption > baseline:
        suggestions.append("Turn off non-essential appliances when not in use.")
    else:
        suggestions.append("Great job! Keep up your energy-saving habits.")
    st.write("**Suggestions:**")
    for s in suggestions:
        st.write(f"- {s}")

# ----------------------------
# DATA EXPORT PAGE
elif page == "Data Export":
    st.title(tr("data_export"))
    st.write("Download your simulated energy consumption data.")
    time, energy_usage = get_simulated_energy_data(24)
    df_export = pd.DataFrame({"Time (Hours)": time, "Energy Consumption (kWh)": energy_usage})
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "energy_data.csv", "text/csv")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Monthly Energy Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Total Energy Consumption: {np.sum(energy_usage):.2f} kWh", ln=True)
        pdf.cell(0, 10, f"Average Hourly Consumption: {np.mean(energy_usage):.2f} kWh", ln=True)
        report_filename = "energy_report.pdf"
        pdf.output(report_filename)
        st.write(f"Download your report: [Energy Report]({report_filename})")

# ----------------------------
# MANUAL INPUT PAGE
elif page == "Manual Input":
    st.title(tr("manual_input"))
    st.write("Enter your own energy consumption data (comma-separated) for analysis.")
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
        st.session_state.manual_data = df_user
    except Exception as e:
        st.error("Invalid input. Please enter numbers separated by commas.")

# ----------------------------
# CHAT ASSISTANT PAGE
elif page == "Chat Assistant":
    st.title(tr("chat_assistant"))
    st.write("Ask for personalized energy-saving tips!")
    question = st.text_input(tr("chat_prompt"), "")
    if st.button(tr("chat_button")) and question:
        # Simulated chatbot response (replace with OpenAI API in production)
        simulated_response = "Tip: Consider upgrading to LED lighting and installing a smart thermostat to optimize energy usage."
        st.write(simulated_response)

# ----------------------------
# IOT INTEGRATION PAGE
elif page == "IoT Integration":
    st.title(tr("iot_integration"))
    st.write("Simulated IoT Sensor Data")
    devices = ["Smart Plug 1", "Smart Plug 2", "Smart Thermostat"]
    sensor_data = {device: np.random.randint(50, 500) for device in devices}
    st.write("Current Sensor Readings (Watts):")
    st.table(pd.DataFrame(list(sensor_data.items()), columns=["Device", "Power (W)"]))
    fig_iot = px.bar(x=list(sensor_data.keys()), y=list(sensor_data.values()),
                     labels={"x": "Device", "y": "Power (W)"}, title="IoT Sensor Readings")
    st.plotly_chart(fig_iot)
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Weather", "Predictions", "Voice Commands"])

# Handle page selection
if page == "Home":
    st.title("Home")
    st.write("Welcome to the app!")
elif page == "Weather":
    st.title("Weather")
    st.write("Weather forecasting coming soon...")
elif page == "Predictions":
    st.title("Predictions")
    st.write("Prediction model coming soon...")
elif page == "Voice Commands":
    voice_commands_page()  # Calls the function from voice_commands.py

