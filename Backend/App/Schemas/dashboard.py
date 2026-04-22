from pydantic import BaseModel
from typing import List, Optional

class SpendingByCategory(BaseModel):
    category: str
    amount: float
    count: int

class SpendingByDepartment(BaseModel):
    department: str
    amount: float

class ApprovalBreakdown(BaseModel):
    status: str
    count: int
    amount: float

class AnomalyByCategory(BaseModel):
    category: str
    count: int

class FinancialOverview(BaseModel):
    total_expense: float
    average_expense: float
    total_transactions: int
    anomaly_count: int
    anomaly_rate: float

class FinancialRatios(BaseModel):
    current_ratio: float
    debt_to_asset: float
    expense_to_revenue: float

class RiskAssessment(BaseModel):
    risk_band: str
    predicted_band: str
    confidence: float
    hmm_state: str

class RiskTimeline(BaseModel):
    fiscal_year: str
    quarter: str
    risk_band: str
    predicted_band: str
    confidence: float
    current_ratio: float
    debt_to_asset: float
    expense_to_revenue: float
    anomaly_rate: float

class RecentTransaction(BaseModel):
    transaction_id: str
    date: Optional[str]
    description: Optional[str]
    category: Optional[str]
    amount: float
    status: Optional[str]
    is_anomaly: bool
    department: Optional[str]

class DashboardResponse(BaseModel):
    overview: FinancialOverview
    financial_ratios: FinancialRatios
    risk_assessment: RiskAssessment
    spending_by_category: List[SpendingByCategory]
    spending_by_department: List[SpendingByDepartment]
    approval_breakdown: List[ApprovalBreakdown]
    anomaly_by_category: List[AnomalyByCategory]
    risk_timeline: List[RiskTimeline]
    recent_transactions: List[RecentTransaction]