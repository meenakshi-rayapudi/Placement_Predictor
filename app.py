import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Placement Predictor",
    layout="wide",
)

# =========================
# 2. LOAD MODEL
# =========================
model = joblib.load("placement_model.pkl")

# =========================
# 3. HEADER
# =========================
st.title("Placement Prediction")
st.write("Predict your placement chances based on your academic and skill profile.")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header(" Your details Input")

age = st.sidebar.number_input("Age", 18, 35)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

degree = st.sidebar.selectbox(
    "Degree",
    ["B.Tech", "BCA", "MCA"]
)

branch = st.sidebar.selectbox(
    "Branch",
    ["Civil", "ECE", "IT", "ME"]
)

cgpa = st.sidebar.number_input(
    "CGPA",
    0.0,
    10.0
)

internships = st.sidebar.number_input(
    "Internships",
    0,
    10
)

projects = st.sidebar.number_input(
    "Projects",
    0,
    10
)

coding = st.sidebar.number_input(
    "Coding Skills",
    0,
    10
)

communication = st.sidebar.number_input(
    "Communication Skills",
    0,
    10
)

aptitude = st.sidebar.number_input(
    "Aptitude Score",
    0,
    100
)

soft_skills = st.sidebar.number_input(
    "Soft Skills",
    0,
    10
)

certifications = st.sidebar.number_input(
    "Certifications",
    0,
    10
)

backlogs = st.sidebar.number_input(
    "Backlogs",
    0,
    10
)

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
st.subheader("Prediction Section")

predict_btn = st.button("Predict Placement")

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

    # =========================
    # RESULT
    # =========================
    if prediction[0] == 1:

        st.success("You are likely to be PLACED")

        st.metric(
            "Placement Confidence",
            f"{probability[0][1]*100:.2f}%"
        )

    else:

        st.error("You are not likely to be Placed")

        st.metric(
            "Placement Confidence",
            f"{probability[0][0]*100:.2f}%"
        )

        # =========================
        # IMPROVEMENT SUGGESTIONS
        # =========================
        st.subheader("Improvement Suggestions")

        suggestions = []

        if cgpa < 7:
            suggestions.append("Improve CGPA")

        if coding < 7:
            suggestions.append("Strengthen coding skills")

        if communication < 7:
            suggestions.append("Improve communication skills")

        if aptitude < 70:
            suggestions.append("Practice aptitude regularly")

        if internships < 1:
            suggestions.append("Complete more internships")

        if projects < 2:
            suggestions.append("Build more real-world projects")

        if certifications < 2:
            suggestions.append("Earn more certifications")

        if backlogs > 0:
            suggestions.append("Clear academic backlogs")

        if len(suggestions) == 0:
            suggestions.append(
                "Profile looks strong. Focus on interview preparation."
            )

        for tip in suggestions:
            st.write(f"✔ {tip}")

    # =========================
    # PROBABILITY CHART
    # =========================
    st.subheader("📊 Prediction Breakdown")

    st.bar_chart({
        "Not Placed": [probability[0][0]],
        "Placed": [probability[0][1]]
    })

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Factors Influencing Prediction")

feature_names = [
    "Age",
    "CGPA",
    "Internships",
    "Projects",
    "Coding Skills",
    "Communication Skills",
    "Aptitude Score",
    "Soft Skills",
    "Certifications",
    "Backlogs",
    "Gender",
    "B.Tech",
    "BCA",
    "MCA",
    "Civil",
    "ECE",
    "IT",
    "ME"
]

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

