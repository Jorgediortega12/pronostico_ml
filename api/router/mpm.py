from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from ..config.database import get_db
from ..services.mpm import MPMService
from ..services.log_service import log_error
from ..services.auth_service import get_current_user
from ..config.schemas import MPMPredictSchema

router = APIRouter(tags=["MPM"])


@router.get("/verify/documents/")
def verify_documents(db: Session = Depends(get_db), service: MPMService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        message = service.verify_documents(db=db)
        return message
    except HTTPException as e:
        log_error(db=db, action="verify_documents", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="verify_documents", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on verify_documents: {str(e)}")
    

@router.get("/demand/{year}/{month}/{previous_days}/")
def get_demand(year: int, month: int, previous_days:int, db: Session = Depends(get_db), service: MPMService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        demand = service.get_demand(year=year, month=month, previous_days=previous_days, db=db)
        return demand
    except HTTPException as e:
        log_error(db=db, action="get_demand_mpm", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_demand_mpm", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_demand: {str(e)}")
    
@router.post("/predict/")
def predict(search: MPMPredictSchema, db: Session = Depends(get_db), service: MPMService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        results = service.predict(search=search, db=db)
        return results
    except HTTPException as e:
        log_error(db=db, action="predict_mpm", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="predict_mpm", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"{str(e)}")


@router.post("/predict/excel/")
def predict_excel(search: MPMPredictSchema, db: Session = Depends(get_db), service: MPMService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        excel_buffer = service.generate_comparison_excel(search=search, db=db)
        
        # Generar nombre de archivo con la fecha
        filename = f"predicciones_mpm_{search.last_date.replace('-', '')}.xlsx"
        
        return StreamingResponse(
            excel_buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except HTTPException as e:
        log_error(db=db, action="predict_excel_mpm", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="predict_excel_mpm", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on predict_excel: {str(e)}")