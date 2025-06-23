import streamlit as st
from PIL import Image
import pandas as pd
from credit_score_app import load_data, clean_data, load_pipeline

image=Image.open("dataset-cover.jpg")
st.image(image,caption="Credit Score", use_container_width=True)
def main():
    st.title("Credit Score Classification")
    
    # Load and prepare data
    df = load_data()
    df, le = clean_data(df)
    pipeline, _ = load_pipeline()
    
    st.header("Input Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        occupation = st.selectbox("Occupation", df['Occupation'].unique())
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0)
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=3)
        num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, value=4)
        interest_rate = st.number_input("Interest Rate", min_value=0, value=10)
        num_loans = st.number_input("Number of Loans", min_value=0, value=2)
        delay_due_date = st.number_input("Delay from Due Date", min_value=-5, value=10)
        num_delayed_payments = st.number_input("Number of Delayed Payments", min_value=0.0, value=2.0)
        
    with col2:
        changed_credit_limit = st.number_input("Changed Credit Limit", value=5.0)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0.0, value=3.0)
        credit_mix = st.selectbox("Credit Mix", df['Credit_Mix'].unique())
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
        credit_util_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, value=30.0)
        payment_min_amount = st.selectbox("Payment of Minimum Amount", df['Payment_of_Min_Amount'].unique())
        total_emi = st.number_input("Total EMI per Month", min_value=0.0, value=100.0)
        amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=200.0)
        payment_behaviour = st.selectbox("Payment Behaviour", df['Payment_Behaviour'].unique())
        monthly_balance = st.number_input("Monthly Balance", min_value=0.0, value=500.0)
    
    if st.button("Predict Credit Score"):
        input_data = {
            'Age': age,
            'Occupation': occupation,
            'Annual_Income': annual_income,
            'Monthly_Inhand_Salary': monthly_salary,
            'Num_Bank_Accounts': num_bank_accounts,
            'Num_Credit_Card': num_credit_cards,
            'Interest_Rate': interest_rate,
            'Num_of_Loan': num_loans,
            'Delay_from_due_date': delay_due_date,
            'Num_of_Delayed_Payment': num_delayed_payments,
            'Changed_Credit_Limit': changed_credit_limit,
            'Num_Credit_Inquiries': num_credit_inquiries,
            'Credit_Mix': credit_mix,
            'Outstanding_Debt': outstanding_debt,
            'Credit_Utilization_Ratio': credit_util_ratio,
            'Payment_of_Min_Amount': payment_min_amount,
            'Total_EMI_per_month': total_emi,
            'Amount_invested_monthly': amount_invested,
            'Payment_Behaviour': payment_behaviour,
            'Monthly_Balance': monthly_balance
        }
        
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        
        st.success(f"Predicted Credit Score: {le.inverse_transform([prediction])[0]}")
        
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Credit Score': le.inverse_transform(range(len(probabilities))),
            'Probability': probabilities
        })
        st.dataframe(prob_df)
        
        st.subheader("Probability Distribution")
        st.bar_chart(prob_df.set_index('Credit Score'))

if __name__ == "__main__":
    main()