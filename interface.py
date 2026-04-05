# ============================================================
# SECTION 1: IMPORTS & PAGE CONFIG
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import time
from pymongo import MongoClient

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# SECTION 2: RESOURCE LOADERS (cached)
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load('printer_model.pkl')

@st.cache_resource
def init_connection():
    client = MongoClient(st.secrets["MONGO_URI"])
    return client["printer_maintenance"]["sensor_data_ml"]
    

# ============================================================
# SECTION 3: DATA FETCHING & PREDICTION
# ============================================================
def fetch_and_predict(collection, model):
    history_data = list(collection.find().sort("_id", -1).limit(1000))

    if not history_data:
        st.warning("No data found in MongoDB. Please run the streaming script first.")
        st.stop()

    df = pd.DataFrame(history_data)
    df.drop(columns=['_id'], inplace=True, errors='ignore')

    features      = df[['air_temp', 'proc_temp', 'rpm', 'torque', 'wear', 'power_w']]
    predictions   = model.predict(features)
    probabilities = model.predict_proba(features)

    df['Model_Prediction'] = ["🚨 FAILURE" if p == 1 else "✅ HEALTHY" for p in predictions]
    df['Confidence']       = [f"{max(prob)*100:.1f}%" for prob in probabilities]

    return df, predictions, probabilities


# ============================================================
# SECTION 4: UI HEADER
# ============================================================
def render_header():
    st.title("🤖 3D Printer AI Predictive Maintenance")
    st.markdown("Real-time sensor monitoring powered by Random Forest Machine Learning.")
    st.markdown("---")


# ============================================================
# SECTION 5: LIVE CHARTS
# ============================================================
def render_charts(df):
    st.subheader("📈 Live Sensor Data Stream")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.caption("🔧 Tool Wear vs Torque")
        st.line_chart(df[['wear', 'torque']].head(50))

    with chart_col2:
        st.caption("⚡ Power vs Air Temp")
        st.line_chart(df[['power_w', 'air_temp']].head(50))


# ============================================================
# SECTION 6: DATA TABLE WITH ROW SELECTION
# ============================================================
def render_table(df):
    st.subheader("📋 Full Dataset AI Analysis — Click any row to inspect")
    display_df = df[[
        'air_temp', 'proc_temp', 'rpm', 'torque',
        'wear', 'power_w', 'status', 'Model_Prediction', 'Confidence'
    ]].reset_index(drop=True)

    selected = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        height=400,
        on_select="rerun",
        selection_mode="single-row"
    )
    return selected


# ============================================================
# SECTION 7: SENSOR METRICS
# ============================================================
def render_metrics(row_data, row_idx):
    st.subheader(f"📊 Sensor Readings — Row {row_idx}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ Air Temp",     f"{row_data['air_temp']} K")
    m2.metric("🔥 Process Temp", f"{row_data['proc_temp']} K",
              delta=round(row_data['proc_temp'] - row_data['air_temp'], 1))
    m3.metric("⚙️ Speed (RPM)",  f"{row_data['rpm']}")
    m4.metric("⚡ Power (W)",    f"{row_data['power_w']} W")

    m5, m6, m7 = st.columns(3)
    m5.metric("🔄 Torque",     f"{row_data['torque']} Nm")
    m6.metric("🛠️ Tool Wear",  f"{row_data['wear']} min")
    m7.metric("📡 Status Tag", row_data['status'])


# ============================================================
# SECTION 8: MODEL DIAGNOSTICS ALERT
# ============================================================
def render_diagnostics(pred_row, conf_row, row_data):
    st.subheader("🧠 Model Diagnostics")
    if pred_row == 1:
        st.error(
            f"🚨 **FAILURE PREDICTED!**\n\n"
            f"**Confidence:** {conf_row:.1f}%\n\n"
            f"**Likely Cause:** {row_data['status']}"
        )
    else:
        st.success(f"✅ **SYSTEM HEALTHY**\n\n**Confidence:** {conf_row:.1f}%")


# ============================================================
# SECTION 9: MAIN APP ORCHESTRATOR
# ============================================================
def main():
    model      = load_model()
    collection = init_connection()

    render_header()

    df, predictions, probabilities = fetch_and_predict(collection, model)

    selected_rows = st.session_state.get("selected_rows", [])
    row_idx  = selected_rows[0] if selected_rows else 0
    row_data = df.iloc[row_idx]
    pred_row = predictions[row_idx]
    conf_row = probabilities[row_idx][pred_row] * 100

    col1, col2 = st.columns([2, 1])
    with col1:
        render_metrics(row_data, row_idx)
    with col2:
        render_diagnostics(pred_row, conf_row, row_data)

    st.markdown("---")

    render_charts(df)

    selected     = render_table(df)
    new_selected = selected.selection.rows
    if new_selected != selected_rows:
        st.session_state["selected_rows"] = new_selected
        st.rerun()

    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()
