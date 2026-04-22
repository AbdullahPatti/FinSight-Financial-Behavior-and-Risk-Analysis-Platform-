from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db import get_db
from Crud.dashboard import get_dashboard_data
from Schemas.dashboard import DashboardResponse

router = APIRouter()

@router.get("/", response_model=DashboardResponse)
def get_dashboard(db: Session = Depends(get_db)):
    """
    Get comprehensive financial dashboard data.
    Returns financial metrics, spending breakdown, risk assessment, and transaction details.
    """
    try:
        data = get_dashboard_data(db)
        return data
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise