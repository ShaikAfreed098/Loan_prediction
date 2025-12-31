import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import numpy as np

# Load datasets
train_df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
test_df = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")

def preprocess_data(df, is_train=True, encoders=None, imputers=None, scaler=None):
    df = df.copy()
    
    # Feature engineering
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['EMI'] = df['EMI'].replace([np.inf, -np.inf], 0)  # replace division errors if any
    
    # Drop columns not needed or redundant
    df = df.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome'], axis=1, errors='ignore')
    
    if is_train:
        y = df['Loan_Status'].map({'Y':1, 'N':0})
        df = df.drop(['Loan_Status'], axis=1)
    else:
        y = None
        test_ids = df['Loan_ID'] if 'Loan_ID' in df.columns else None
        if 'Loan_ID' in df.columns:
            df = df.drop(['Loan_ID'], axis=1)
    
    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Impute numeric missing
    if is_train:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
    else:
        df[num_cols] = imputers['num'].transform(df[num_cols])
    
    # Impute categorical missing
    if is_train:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    else:
        df[cat_cols] = imputers['cat'].transform(df[cat_cols])
    
    # Label encode categoricals
    if is_train:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Scale features
    if is_train:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = scaler.transform(df)
    
    if is_train:
        imputers = {'num': num_imputer, 'cat': cat_imputer}
        return df_scaled, y, encoders, imputers, scaler
    else:
        return df_scaled, test_ids

# Preprocess data
X_train, y_train, encoders, imputers, scaler = preprocess_data(train_df, is_train=True)
X_test, test_ids = preprocess_data(test_df, is_train=False, encoders=encoders, imputers=imputers, scaler=scaler)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Hyperparameter grid to tune
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1]
}

# Grid search with 3-fold CV
grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_tr, y_tr)

print("Best hyperparameters:", grid_search.best_params_)

# Evaluate on validation
val_preds = grid_search.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Classification Report:\n", classification_report(y_val, val_preds))

# Predict on test
test_preds = grid_search.predict(X_test)
test_preds_labels = pd.Series(test_preds).map({1:'Y', 0:'N'})

# Save submission
submission = pd.DataFrame({'Loan_ID': test_ids, 'Loan_Status': test_preds_labels})
submission.to_csv('loan_submission_improved.csv', index=False)
print("Improved submission saved as loan_submission_improved.csv")


# === TRAINING PART ===
# Example: XGBoost
from xgboost import XGBClassifier
import joblib

# X_train, y_train already prepared
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Suppose you also have encoders, imputers, scaler
# encoders = {...}, imputers = {...}, scaler = scaler

# === STEP 2: SAVE MODEL AND PREPROCESSORS ===
joblib.dump(model, 'loan_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(imputers, 'imputers.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and preprocessing objects saved successfully!")
