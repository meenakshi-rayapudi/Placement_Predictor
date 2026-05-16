Placement Prediction System

An interactive machine learning web application that predicts student placement chances based on academic performance, skills, internships, projects, and certifications.

Built using Python, Streamlit, and machine learning.

->Features

Placement prediction using Random Forest  
Placement probability score  
Personalized improvement suggestions  
Feature importance visualization  
Interactive sidebar UI  


->Tech Stack

-Python
-Streamlit
-NumPy
-Pandas
-Scikit-learn
-Joblib


->Project Structure

Placement Predictor/
│
├── data/
│   └── placements.csv
│
├── notebook/
│   └── eda.ipynb
│
├── placement_model.pkl
├── app.py
├── req.txt
└── README.md


-> Dataset Features

The model uses:
-Age
-CGPA
-Internships
-Projects
-Coding Skills
-Communication Skills
-Aptitude Score
-Soft Skills
-Certifications
-Backlogs
-Gender
-Degree
-Branch


->Machine Learning Model

Model used:Random Forest Classifier

because:
-Handles nonlinear relationships
-Good accuracy
-Provides feature importance
-Works well on mixed data


->How to Run

Install dependencies: pip install -r req.txt

Run the app: streamlit run app.py


->Application Modules

Placement Prediction
Predicts whether a student is likely to be placed.

Improvement Suggestions
Provides personalized recommendations.

Feature Importance
Shows which features influence predictions.


