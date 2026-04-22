from pydantic import BaseModel
from datetime import date
from typing import Optional

class TransactionResponse(BaseModel):
    transaction_id: str
    date: date
    department: Optional[str]
    expense_category: Optional[str]
    amount_pkr: float
    payment_method: Optional[str]
    approval_status: Optional[str]
    hmm_state: Optional[str]
    is_anomaly: bool
    review_tier: Optional[str]
    predicted_category: Optional[str]

    class Config:
        from_attributes = True

class SingleExpenseInput(BaseModel):
    department: Optional[str] = None
    expense_category: Optional[str] = None
    vendor_name: Optional[str] = None
    transaction_description: str
    amount_pkr: float
    payment_method: Optional[str] = None
    approval_status: Optional[str] = "Pending"
    fiscal_year: int
    quarter: int
    current_assets_pkr: float
    current_liabilities_pkr: float
    fixed_assets_pkr: float
    total_assets_pkr: float
    total_liabilities_pkr: float
    long_term_loans_pkr: float
    quarterly_revenue_pkr: float
