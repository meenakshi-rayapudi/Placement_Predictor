import streamlit as st
import numpy as np
import joblib

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Placement Predictor",
    layout="wide",
    page_icon="🎓"
)

# =========================
# 2. LOAD MODEL
# =========================
model = joblib.load("placement_model.pkl")

# =========================
# 3. HEADER
# =========================
st.title("🎓 AI Placement Prediction System")
st.write("Predict whether a student will be placed based on academic & skill profile")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📌 Student Details Input")

age = st.sidebar.number_input("Age", 18, 35)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

degree = st.sidebar.selectbox("Degree", ["B.Tech", "BCA", "MCA"])

branch = st.sidebar.selectbox("Branch", ["Civil", "ECE", "IT", "ME"])

cgpa = st.sidebar.number_input("CGPA", 0.0, 10.0)

internships = st.sidebar.number_input("Internships", 0, 10)

projects = st.sidebar.number_input("Projects", 0, 10)

coding = st.sidebar.number_input("Coding Skills", 0, 10)

communication = st.sidebar.number_input("Communication Skills", 0, 10)

aptitude = st.sidebar.number_input("Aptitude Score", 0, 100)

soft_skills = st.sidebar.number_input("Soft Skills", 0, 10)

certifications = st.sidebar.number_input("Certifications", 0, 10)

backlogs = st.sidebar.number_input("Backlogs", 0, 10)

# =========================
# ENCODING
# =========================
gender_male = 1 if gender == "Male" else 0

degree_btech = 1 if degree == "B.Tech" else 0
degree_bca = 1 if degree == "BCA" else 0
degree_mca = 1 if degree == "MCA" else 0

branch_civil = 1 if branch == "Civil" else 0
branch_ece = 1 if branch == "ECE" else 0
branch_it = 1 if branch == "IT" else 0
branch_me = 1 if branch == "ME" else 0

# =========================
# MAIN PAGE BUTTON
# =========================
st.subheader("🔍 Prediction Section")

col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button("🚀 Predict Placement")

# =========================
# PREDICTION
# =========================
if predict_btn:

    input_data = np.array([[
        age,
        cgpa,
        internships,
        projects,
        coding,
        communication,
        aptitude,
        soft_skills,
        certifications,
        backlogs,
        gender_male,
        degree_btech,
        degree_bca,
        degree_mca,
        branch_civil,
        branch_ece,
        branch_it,
        branch_me
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.markdown("---")

    if prediction[0] == 1:
        st.success("🎉 Student is LIKELY to be PLACED")
        st.metric("Placement Probability", f"{probability[0][1]*100:.2f}%")
    else:
        st.error("⚠ Student is NOT likely to be placed")
        st.metric("Placement Probability", f"{probability[0][0]*100:.2f}%")

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📊 Prediction Breakdown")

    st.bar_chart({
        "Not Placed": [probability[0][0]],
        "Placed": [probability[0][1]]
    })

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit • Placement Prediction ML Project")