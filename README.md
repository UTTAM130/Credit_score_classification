# Credit Score Classification

This project builds a machine learning model to classify credit scores ("Good," "Standard," or "Poor") based on customer credit-related information. It uses an XGBoost classifier with a preprocessing pipeline to handle numerical and categorical features. The application includes a FastAPI service for programmatic predictions and a Streamlit interface for user-friendly input and visualization.

## Project Structure

- `credit_score_app.py`: Core logic for data loading, cleaning, preprocessing, and model training.
- `api.py`: FastAPI service providing a `/predict` endpoint for credit score predictions.
- `streamlit_app.py`: Streamlit web interface for interactive user input and result visualization.
- `credit_score_classification.csv`: Input dataset (not included in repository; user must provide).
- `credit_score_model.pkl`: Trained model file (generated after running `credit_score_app.py`).

## Requirements

- Python 3.8+
- Required packages:
  ```bash
  pip install pandas numpy scikit-learn xgboost fastapi pydantic streamlit uvicorn
  ```

## Setup Instructions

1. **Clone or Download the Repository**
   - Ensure all three Python files (`credit_score_app.py`, `api.py`, `streamlit_app.py`) are in the same directory.

2. **Place the Dataset**
   - Add the `credit_score_classification.csv` file to the project directory. This file should contain the credit-related data with the structure described in the dataset section below.

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or install directly:
   ```bash
   pip install pandas numpy scikit-learn xgboost fastapi pydantic streamlit uvicorn
   ```

4. **Train the Model**
   ```bash
   python credit_score_app.py
   ```
   This generates `credit_score_model.pkl` containing the trained XGBoost model and label encoder.

## Running the Application

1. **Start the FastAPI Server**
   In a terminal, run:
   ```bash
   python api.py
   ```
   - Access the API documentation at: `http://localhost:8000/docs`
   - Use the `/predict` endpoint to make predictions via POST requests.

2. **Start the Streamlit Interface**
   In a separate terminal, run:
   ```bash
   streamlit run streamlit_app.py
   ```
   - Open the web interface at: `http://localhost:8501`
   - Enter customer information and click "Predict Credit Score" to view results.

## Dataset

The `credit_score_classification.csv` file should contain 100,000 entries with 28 columns, including:
- Numerical features: `Monthly_Inhand_Salary`, `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Delay_from_due_date`, `Num_Credit_Inquiries`, `Credit_Utilization_Ratio`, `Total_EMI_per_month`, `Age`, `Annual_Income`, `Num_of_Loan`, `Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Outstanding_Debt`, `Amount_invested_monthly`, `Monthly_Balance`
- Categorical features: `Occupation`, `Credit_Mix`, `Payment_of_Min_Amount`, `Payment_Behaviour`
- Target: `Credit_Score` ("Good," "Standard," or "Poor")
- Ignored columns: `ID`, `Customer_ID`, `Month`, `Name`, `SSN`, `Type_of_Loan`, `Credit_History_Age`

The model handles missing values and mixed types through imputation and preprocessing.

## Usage Examples

### Streamlit Interface
1. Navigate to `http://localhost:8501`.
2. Input customer details (e.g., Age: 35, Occupation: Doctor, Annual_Income: 120000, etc.).
3. Click "Predict Credit Score" to see the predicted score and probability distribution.

### FastAPI Endpoint
Send a POST request to `http://localhost:8000/predict` with a JSON payload like:
```json
{
  "Age": 35,
  "Occupation": "Doctor",
  "Annual_Income": 120000.0,
  "Monthly_Inhand_Salary": 10000.0,
  "Num_Bank_Accounts": 2,
  "Num_Credit_Card": 3,
  "Interest_Rate": 5,
  "Num_of_Loan": 1,
  "Delay_from_due_date": 3,
  "Num_of_Delayed_Payment": 0.0,
  "Changed_Credit_Limit": 2.5,
  "Num_Credit_Inquiries": 1.0,
  "Credit_Mix": "Good",
  "Outstanding_Debt": 500.0,
  "Credit_Utilization_Ratio": 25.0,
  "Payment_of_Min_Amount": "No",
  "Total_EMI_per_month": 150.0,
  "Amount_invested_monthly": 2000.0,
  "Payment_Behaviour": "High_spent_Large_value_payments",
  "Monthly_Balance": 5000.0
}
```
Example using `curl`:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @input.json
```

Response:
```json
{
  "credit_score": "Good",
  "probabilities": {
    "Good": 0.85,
    "Poor": 0.05,
    "Standard": 0.10
  }
}
```

## Troubleshooting

- **DtypeWarning**: Ensure `credit_score_classification.csv` is clean. Mixed types in columns like `Monthly_Balance` are handled, but verify data consistency.
- **Model Training Error**: Check that the dataset has the expected columns and valid values. Run `python credit_score_app.py` to regenerate the model.
- **API/Streamlit Not Loading**: Confirm ports 8000 (FastAPI) and 8501 (Streamlit) are free. Kill conflicting processes or change ports in the scripts.
- **Invalid Categorical Values**: Ensure inputs for `Occupation`, `Credit_Mix`, `Payment_of_Min_Amount`, and `Payment_Behaviour` match the dataset's unique values.

## Notes

- The model uses XGBoost with a preprocessing pipeline that includes `OneHotEncoder` for categorical features and `SimpleImputer` for missing values.
- Performance may vary with high-cardinality categorical features due to `OneHotEncoder`. Consider `OrdinalEncoder` for optimization if needed.
- The project was tested with Python 3.8+ on Windows and Unix systems as of June 23, 2025.

## License

This project is licensed under the MIT License.

## Contact

For issues or questions, please open an issue on the repository or contact the maintainer.