import streamlit as st
import pandas as pd
import joblib
import time
from pymongo import MongoClient

# --- Page Config ---
st.set_page_config(page_title="AI Predictive Maintenance", page_icon="⚙️", layout="wide")

# --- 1. Load ML Model ---
@st.cache_resource
def load_model():
    return joblib.load('printer_model.pkl')

model = load_model()

# --- 2. MongoDB Connection ---
@st.cache_resource
def init_connection():
    client = MongoClient("mongodb+srv://databasesensor:1234mongo@navdeep.p32wk6a.mongodb.net/?appName=navdeep")
    return client["printer_maintenance"]["sensor_data_ml"]

collection = init_connection()

# --- UI Header ---
st.title("🤖 3D Printer AI Predictive Maintenance")
st.markdown("Real-time sensor monitoring powered by Random Forest Machine Learning.")
st.markdown("---")

# --- Fetch Data ---
history_data = list(collection.find().sort("_id", -1).limit(1000))

if not history_data:
    st.warning("No data found in MongoDB. Please run the streaming script first.")
    st.stop()

df = pd.DataFrame(history_data)
df.drop(columns=['_id'], inplace=True, errors='ignore')

# --- Batch Predictions on full df ---
features_batch = df[['air_temp', 'proc_temp', 'rpm', 'torque', 'wear', 'power_w']]
predictions    = model.predict(features_batch)
probabilities  = model.predict_proba(features_batch)

df['Model_Prediction'] = ["🚨 FAILURE" if p == 1 else "✅ HEALTHY" for p in predictions]
df['Confidence']       = [f"{max(prob)*100:.1f}%" for prob in probabilities]

display_df = df[['air_temp', 'proc_temp', 'rpm', 'torque', 'wear', 'power_w', 'status', 'Model_Prediction', 'Confidence']].reset_index(drop=True)

# --- Layout ---
col1, col2 = st.columns([2, 1])

# --- Live Charts ABOVE table ---
st.subheader("📈 Live Sensor Data Stream")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.caption("🔧 Tool Wear vs Torque")
    st.line_chart(df[['wear', 'torque']].head(50))
with chart_col2:
    st.caption("⚡ Power vs Air Temp")
    st.line_chart(df[['power_w', 'air_temp']].head(50))

# --- Table with row selection ---
st.subheader("📋 Full Dataset AI Analysis — Click any row to inspect")
selected = st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=False,
    height=400,
    on_select="rerun",
    selection_mode="single-row"
)

# --- Determine which row to show in metrics/alert ---
selected_rows = selected.selection.rows
if selected_rows:
    row_idx  = selected_rows[0]
    row_data = df.iloc[row_idx]
else:
    # Default to latest row if no selection
    row_idx  = 0
    row_data = df.iloc[0]

pred_row = predictions[row_idx]
conf_row = probabilities[row_idx][pred_row] * 100

# --- Metrics ---
with col1:
    st.subheader(f"📊 Sensor Readings — Row {row_idx}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ Air Temp",     f"{row_data['air_temp']} K")
    m2.metric("🔥 Process Temp", f"{row_data['proc_temp']} K",
              delta=round(row_data['proc_temp'] - row_data['air_temp'], 1))
    m3.metric("⚙️ Speed (RPM)",  f"{row_data['rpm']}")
    m4.metric("⚡ Power (W)",    f"{row_data['power_w']} W")

    m4, m5, m6 = st.columns(3)
    m4.metric("🔄 Torque",     f"{row_data['torque']} Nm")
    m5.metric("🛠️ Tool Wear",  f"{row_data['wear']} min")
    m6.metric("📡 Status Tag", row_data['status'])

# --- Alert ---
with col2:
    st.subheader("🧠 Model Diagnostics")
    if pred_row == 1:
        st.error(f"🚨 **FAILURE PREDICTED!**\n\n**Confidence:** {conf_row:.1f}%\n\n**Likely Cause:** {row_data['status']}")
    else:
        st.success(f"✅ **SYSTEM HEALTHY**\n\n**Confidence:** {conf_row:.1f}%")

# --- Auto refresh every 5 seconds ---
time.sleep(5)
st.rerun()