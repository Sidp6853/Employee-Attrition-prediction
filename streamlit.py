import streamlit as st
import numpy as np
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("emp_attrition_model.pkl")
    return model

model = load_model()

# Categorical encodings (must match training!)
department_map = {
    "sales": 0,
    "accounting": 1,
    "hr": 2,
    "technical": 3,
    "support": 4,
    "management": 5,
    "IT": 6,
    "product_mng": 7,
    "marketing": 8,
    "RandD": 9
}

salary_map = {
    "low": 0,
    "medium": 1,
    "high": 2
}

st.title("👩‍💼 Employee Attrition Prediction")
st.write("Predict whether an employee is likely to leave the company based on their profile.")

st.sidebar.header("Employee Details")

# Numeric inputs
satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01)
last_evaluation = st.sidebar.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.01)
number_project = st.sidebar.number_input("Number of Projects", min_value=1, max_value=10, value=3, step=1)
average_montly_hours = st.sidebar.number_input("Average Monthly Hours", min_value=50, max_value=350, value=160, step=1)
time_spend_company = st.sidebar.number_input("Years at Company", min_value=1, max_value=20, value=3, step=1)
Work_accident = st.sidebar.selectbox("Had Work Accident?", ["No", "Yes"])
promotion_last_5years = st.sidebar.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])

# Categorical inputs
Department = st.sidebar.selectbox(
    "Department",
    ["sales", "accounting", "hr", "technical", "support",
     "management", "IT", "product_mng", "marketing", "RandD"]
)
salary = st.sidebar.selectbox("Salary Level", ["low", "medium", "high"])

# Convert Yes/No to 0/1
Work_accident_bin = 1 if Work_accident == "Yes" else 0
promotion_bin = 1 if promotion_last_5years == "Yes" else 0

# Encode categorical features
dept_encoded = department_map[Department]
salary_encoded = salary_map[salary]

# Prepare feature vector
features = np.array([[  
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    Work_accident_bin,
    promotion_bin,
    dept_encoded,
    salary_encoded
]])

# st.subheader("⚙ Input Summary")
# st.write({
#     "satisfaction_level": satisfaction_level,
#     "last_evaluation": last_evaluation,
#     "number_project": number_project,
#     "average_montly_hours": average_montly_hours,
#     "time_spend_company": time_spend_company,
#     "Work_accident": Work_accident_bin,
#     "promotion_last_5years": promotion_bin,
#     "Department": Department,
#     "salary": salary,
# })

if st.button("🔮 Predict Attrition"):
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if pred == 1:
        st.error(f"⚠ Employee is likely to **LEAVE**. (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Employee is likely to **STAY**. (Probability of leaving: {proba:.2f})")

    st.caption("Note: Probability is the model's estimated chance of the employee leaving.")
