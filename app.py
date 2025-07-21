import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("/home/rguktrkvalley/Documents/Loan-Prediction-System/train.csv")
    df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean(), inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mean(), inplace=True)
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    return df

def train_model(df):
    df = df.drop(columns=["Loan_ID"])
    cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status", "Dependents"]
    for col in cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop(columns="Loan_Status")
    y = df["Loan_Status"]
    model = LogisticRegression()
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

st.title("Loan Prediction System")
st.write("Enter applicant details to predict loan approval:")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

input_dict = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 0 if education == "Graduate" else 1,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.subheader(result)

