from fastapi import APIRouter, UploadFile, File, Depends
import pandas as pd
import numpy as np
import joblib
import re
from sqlalchemy.orm import Session
from db import get_db
from Core.pipelines import run_full_pipeline
from Crud.transaction import bulk_insert_transactions
from Schemas.transaction import SingleExpenseInput
from Crud.quarterly import bulk_insert_quarterly

router = APIRouter()

@router.post("/")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        if not file.filename.endswith(".csv"):
            return {"error": "Only CSV files are allowed"}
        
        print(f"\nReceived file: {file.filename}")
        content = await file.read()
        print(f"File size: {len(content)} bytes")
        
        with open("temp_upload.csv", "wb") as f:
            f.write(content)
        print("File saved as temp_upload.csv")
        
        success = run_full_pipeline("temp_upload.csv")
        
        if success:
            try:
                print("\nInserting transactions into database...")
                bulk_insert_transactions(db)
                print("Transactions inserted successfully!")
                
                print("Inserting quarterly summary into database...")
                bulk_insert_quarterly(db)
                print("Quarterly summary inserted successfully!")
                
                return {"message": "File uploaded and processed successfully", "status": "success"}
            except Exception as db_error:
                print(f"Database insertion error: {str(db_error)}")
                import traceback
                print(traceback.format_exc())
                return {"error": f"Database error: {str(db_error)}"}
        else:
            return {"error": "Pipeline processing failed - check server logs for details"}
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"error": f"Upload failed: {str(e)}"}

def analyze_single_expense(expense: SingleExpenseInput):
    df = pd.DataFrame([expense.dict()])
    try:
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        model = joblib.load('nlp_logreg.pkl')
        le = joblib.load('label_encoder.pkl')
        desc_clean = df['transaction_description'].str.lower().str.replace(r'[^a-z\s]', ' ', regex=True)
        X = tfidf.transform(desc_clean)
        predicted = model.predict(X)
        predicted_category = le.inverse_transform(predicted)[0]
        return {"predicted_category": predicted_category}
    except FileNotFoundError:
        return {"error": "Models not found, please upload a CSV file first to train the models"}

@router.post("/single")
async def analyze_single(expense: SingleExpenseInput):
    return analyze_single_expense(expense)
