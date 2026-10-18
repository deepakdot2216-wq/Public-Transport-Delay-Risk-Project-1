
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Bus Delay Risk Dashboard",
    page_icon="🚌",
    layout="wide"
)

# Load trained model
model = joblib.load("delay_risk_model.pkl")
features = joblib.load("feature_columns.pkl")

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #142b46);
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
h1, h2, h3, p, label {
    color: #f1f5f9 !important;
}
.hero {
    background: linear-gradient(120deg, #17365d, #126e82);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #315777;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
}
.hero p {
    color: #d6e8f5 !important;
    font-size: 16px;
}
.panel {
    background: #17283d;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #30465f;
    margin-bottom: 18px;
}
div[data-testid="stMetric"] {
    background: #17283d;
    border: 1px solid #30465f;
    padding: 18px;
    border-radius: 14px;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #06b6d4, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: bold;
    min-height: 48px;
}
.stButton > button:hover {
    border: 1px solid #67e8f9;
    color: white;
}
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div {
    border-radius: 10px;
}
.result {
    padding: 24px;
    border-radius: 16px;
    margin-top: 20px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.high {
    background: #451c27;
    border: 1px solid #ef6474;
    color: #ffb4bd;
}
.low {
    background: #123b35;
    border: 1px solid #34d399;
    color: #a7f3d0;
}
.footer {
    text-align: center;
    color: #94a3b8;
    padding-top: 25px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# Dashboard header
st.markdown("""
<div class="hero">
    <h1>🚌 Public Transport Delay Risk</h1>
    <p>Intelligent bus incident analysis and machine learning prediction dashboard</p>
    <p>📍 NYC Bus Operations &nbsp; | &nbsp; 🤖 ML-Powered Prediction</p>
</div>
""", unsafe_allow_html=True)

# Top metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("System Status", "Ready", "Model Loaded")

with col2:
    st.metric("Prediction Type", "Classification")

with col3:
    st.metric("Transport Mode", "School Buses")

st.write("")

# Input section
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📋 Bus Incident Details")
    st.write("Enter the details below to predict the delay risk.")

    with st.form("prediction_form"):
        run_type = st.selectbox(
            "Run Type", ["AM Run", "PM Run"]
        )

        boro = st.selectbox(
            "Borough",
            ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
        )

        breakdown = st.selectbox(
            "Incident Type", ["Breakdown", "Running Late"]
        )

        school_age = st.selectbox(
            "School Category", ["School-Age", "Pre-K"]
        )

        students = st.number_input(
            "Number of Students",
            min_value=0,
            value=20,
            step=1
        )

        submitted = st.form_submit_button("🔍 Predict Delay Risk")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Overview")
    st.write("Your prediction will be based on the selected bus information.")

    st.markdown("**Selected borough**")
    st.info("📍 " + boro)

    st.markdown("**Bus schedule**")
    st.info("🕒 " + run_type)

    st.markdown("**Incident category**")
    st.info("⚙️ " + breakdown)

    st.markdown("**Students on board**")
    st.metric("Student Count", students)

    st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if submitted:
    data = pd.DataFrame({
        "Run_Type": [run_type],
        "Boro": [boro],
        "Breakdown_or_Running_Late": [breakdown],
        "School_Age_or_PreK": [school_age],
        "Number_Of_Students_On_The_Bus": [students]
    })

    data = pd.get_dummies(data)
    data = data.reindex(columns=features, fill_value=0)

    prediction = model.predict(data)

    st.subheader("🎯 Prediction Result")

    if prediction[0] == 1:
        st.markdown("""
        <div class="result high">
            ⚠️ HIGH DELAY RISK
            <p style="font-size:15px;font-weight:normal;">
            This incident is classified as high risk by the trained model.
            Consider monitoring the bus situation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result low">
            ✅ LOW DELAY RISK
            <p style="font-size:15px;font-weight:normal;">
            This incident is classified as low risk by the trained model.
            Continue normal monitoring.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Note: The result is a model prediction, not a guarantee of actual bus performance."
    )

# Footer
st.markdown("""
<div class="footer">
    Public Transport Delay Risk Prediction System<br>
    Developed by Deepak Prajapati | Machine Learning Project
</div>
""", unsafe_allow_html=True)
