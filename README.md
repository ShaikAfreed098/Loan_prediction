# Loan Eligibility Prediction using Machine Learning

## Project Overview
This project predicts whether an applicant is eligible for a loan based on various personal and financial details using machine learning models. The system includes a simple web interface where users can input their data and get real-time loan eligibility predictions.

---

## Dataset
We use the **Loan Prediction Dataset** from Kaggle. The dataset is split into:

1. `train_u6lujuX_CVtuZ9i.csv` – Training dataset with known loan outcomes.
2. `test_Y3wMUE5_7gLdaTN.csv` – Test dataset for making predictions.

### Key Columns
- **Loan_ID**: Unique Loan ID
- **Gender**: Male / Female
- **Married**: Yes / No
- **Dependents**: Number of dependents
- **Education**: Graduate / Not Graduate
- **Self_Employed**: Yes / No
- **ApplicantIncome**: Applicant's monthly income
- **CoapplicantIncome**: Coapplicant's income
- **LoanAmount**: Loan amount (in thousands)
- **Loan_Amount_Term**: Term of loan in months
- **Credit_History**: 1 if meets guidelines, 0 otherwise
- **Property_Area**: Urban / Semiurban / Rural
- **Loan_Status**: Y (Approved) / N (Not Approved)

---

## Machine Learning Model
- **Model Used**: XGBoost Classifier
- **Accuracy**: > 90% on training data
- **Preprocessing**:
  - Handling missing values
  - Encoding categorical variables
  - Scaling numerical features if necessary

---



Loan_prediction/
├── lep.py # ML training and model creation script
├── app.py # Flask app for web interface
├── train_u6lujuX_CVtuZ9i.csv # Training dataset
├── test_Y3wMUE5_7gLdaTN.csv # Test dataset
├── loan_model.pkl # Trained ML model
├── README.md # Project documentation
└── requirements.txt # Python dependencies




---

## How to Run

### 1. Install Dependencies

pip install -r requirements.txt


Features

User-friendly web interface

Real-time prediction using trained ML model

Handles missing values and categorical variables

High prediction accuracy

Tools & Technologies

Python 3.x

Pandas, NumPy

Scikit-learn

XGBoost

Flask (for web interface)

Joblib (for saving/loading model)


Future Improvements

Deploy the app online using Heroku or Streamlit

Include more features like previous loan defaults, employment type, etc.

Add visualization of feature importance

Integrate authentication for user privacy

#Author

#Shaik Afreed
#Email: shaikafreed@example.com

GitHub: https://github.com/ShaikAfreed098## Project Structure

