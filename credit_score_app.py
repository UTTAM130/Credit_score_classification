import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import pickle
import re

def load_data(file_path='credit_score_classification.csv'):
    """Load and return the dataset."""
    return pd.read_csv(file_path, low_memory=False)

def clean_data(df):
    """Clean and preprocess the dataset."""
    numeric_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                    'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 
                    'Monthly_Balance']
    
    for col in numeric_cols:
        # Remove non-numeric characters and convert to numeric
        df[col] = pd.to_numeric(df[col].astype(str).replace(r'[^\d.]+', '', regex=True), errors='coerce')
    
    # Clip outliers
    df['Age'] = df['Age'].clip(lower=0, upper=100)
    df['Annual_Income'] = df['Annual_Income'].clip(lower=0)
    
    # Handle mixed types in Monthly_Balance explicitly
    df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')
    
    # Encode target variable
    le = LabelEncoder()
    df['Credit_Score'] = le.fit_transform(df['Credit_Score'])
    
    return df, le

def create_preprocessor():
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_features = ['Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                       'Interest_Rate', 'Delay_from_due_date', 'Num_Credit_Inquiries',
                       'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Age',
                       'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                       'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly',
                       'Monthly_Balance']
    
    categorical_features = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 
                          'Payment_Behaviour']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

def train_model(df, le, model_path='credit_score_model.pkl'):
    """Train and save the XGBoost model."""
    X = df.drop(['Credit_Score', 'ID', 'Customer_ID', 'Month', 'Name', 'SSN', 
                 'Type_of_Loan', 'Credit_History_Age'], axis=1)
    y = df['Credit_Score']
    
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('classifier', XGBClassifier(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    with open(model_path, 'wb') as f:
        pickle.dump({'model': pipeline, 'label_encoder': le}, f)
    
    return pipeline, le

def load_pipeline(model_path='credit_score_model.pkl'):
    """Load the trained model and label encoder."""
    with open(model_path, 'rb') as f:
        saved_model = pickle.load(f)
    return saved_model['model'], saved_model['label_encoder']

if __name__ == "__main__":
    df = load_data()
    df, le = clean_data(df)
    pipeline, le = train_model(df, le)
    print("Model trained and saved successfully.")