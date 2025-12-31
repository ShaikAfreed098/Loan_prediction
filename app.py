import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and preprocessing objects
model = joblib.load('loan_model.pkl')
encoders = joblib.load('encoders.pkl')
imputers = joblib.load('imputers.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Loan Eligibility Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create a dataframe for model
input_df = pd.DataFrame({
    'Gender':[gender],
    'Married':[married],
    'Dependents':[dependents],
    'Education':[education],
    'Self_Employed':[self_employed],
    'ApplicantIncome':[applicant_income],
    'CoapplicantIncome':[coapplicant_income],
    'LoanAmount':[loan_amount],
    'Loan_Amount_Term':[loan_term],
    'Credit_History':[credit_history],
    'Property_Area':[property_area]
})

# Preprocess features the same way as training
input_df['Dependents'] = input_df['Dependents'].replace('3+', 3).astype(float)
input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
input_df['EMI'] = input_df['LoanAmount'] / input_df['Loan_Amount_Term']
input_df['EMI'] = input_df['EMI'].replace([np.inf, -np.inf], 0)
input_df = input_df.drop(['ApplicantIncome','CoapplicantIncome'], axis=1)

# Impute numeric and categorical missing values
num_cols = input_df.select_dtypes(include=['float64','int64']).columns
cat_cols = input_df.select_dtypes(include=['object']).columns
input_df[num_cols] = imputers['num'].transform(input_df[num_cols])
input_df[cat_cols] = imputers['cat'].transform(input_df[cat_cols])

# Encode categoricals
for col in cat_cols:
    le = encoders[col]
    input_df[col] = input_df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
    input_df[col] = le.transform(input_df[col])

# Scale numeric features
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Check Eligibility"):
    pred = model.predict(input_scaled)
    if pred[0] == 1:
        st.success("Congratulations! You are eligible for the loan.")
    else:
        st.error("Sorry! You are not eligible for the loan.")
