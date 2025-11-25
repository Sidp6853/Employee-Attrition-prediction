from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib


model = joblib.load("emp_attrition_model.pkl")

app = FastAPI()


class EmployeeData(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int
    Department: str
    salary: str



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


@app.get("/")
def home():
    return {"message": "Attrition Prediction Model is running!"}


@app.post("/predict")
def predict(data: EmployeeData):

    
    dept_encoded = department_map.get(data.Department, -1)
    salary_encoded = salary_map.get(data.salary, -1)

    if dept_encoded == -1 or salary_encoded == -1:
        return {"error": "Invalid Department or Salary value."}

    
    input_features = np.array([[  
        data.satisfaction_level,
        data.last_evaluation,
        data.number_project,
        data.average_montly_hours,
        data.time_spend_company,
        data.Work_accident,
        data.promotion_last_5years,
        dept_encoded,
        salary_encoded
    ]])

    
    pred = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0][1]

    return {
        "left": int(pred),
        "left_probability": float(proba)
    }
