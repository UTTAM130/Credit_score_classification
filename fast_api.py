from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from credit_score_app import load_pipeline

app = FastAPI()

class CreditInput(BaseModel):
    Age: float
    Occupation: str
    Annual_Income: float
    Monthly_Inhand_Salary: Optional[float]
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: int
    Num_of_Loan: int
    Delay_from_due_date: int
    Num_of_Delayed_Payment: Optional[float]
    Changed_Credit_Limit: Optional[float]
    Num_Credit_Inquiries: Optional[float]
    Credit_Mix: str
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Payment_of_Min_Amount: str
    Total_EMI_per_month: float
    Amount_invested_monthly: Optional[float]
    Payment_Behaviour: str
    Monthly_Balance: Optional[float]

@app.post("/predict")
async def predict_credit_score(input_data: CreditInput):
    try:
        pipeline, le = load_pipeline()
        
        input_df = pd.DataFrame([input_data.dict()])
        
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]
        
        return {
            "credit_score": le.inverse_transform([prediction])[0],
            "probabilities": {
                le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(probability)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)