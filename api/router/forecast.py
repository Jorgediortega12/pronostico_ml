from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from ..config.database import get_db
from ..services.forecast import ForecastService
from ..services.log_service import log_error
from ..services.auth_service import get_current_user
from ..config.schemas import ForecastSchema, NewUserModelSchema, CreateOrUpdateUsersModelsSchema, MonthlyInfoSchema, DayBehaviorSchema, CreateTypeYearListSchema, UpdateTypeYearListSchema, UpdateMonthlyTypeSchema, ForecastTypeMonthSchema, ForecastModelSaveBasedOnYear
from datetime import datetime
import traceback
router = APIRouter(tags=["Forecast"])

@router.post("/predict/")
def predict(search: ForecastSchema, db: Session = Depends(get_db),  service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        predictions = service.predict(search=search, db=db)
        return predictions
    except HTTPException as e:
        log_error(db=db, action="predict", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="predict", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on predict: {str(e)}")


@router.get("/demand/monthly/")
def get_monthly_demand(db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        demand = service.get_monthly_percentage(db=db)
        return demand
    except HTTPException as e:
        log_error(db=db, action="get_monthly_demand", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_monthly_demand", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_monthly_demand: {str(e)}")

@router.post("/model/create/")
def create_model(search: NewUserModelSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        model = service.create_new_model(search=search, db=db)
        return model
    except HTTPException as e:
        log_error(db=db, action="create_model", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="create_model", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on create_model: {str(e)}")

@router.post("/model/savevalues/")
def create_update_user_model(search: CreateOrUpdateUsersModelsSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        model = service.create_or_update_user_model(search=search, db=db)
        return model
    except HTTPException as e:
        log_error(db=db, action="create_update_user_model", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="create_update_user_model", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on create_update_user_model: {str(e)}")
    
@router.get("/model/values/{model_id}/")
def get_user_model_values(model_id: int, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        values = service.retrieve_user_model_values(model_id=model_id, db=db)
        return values
    except HTTPException as e:
        log_error(db=db, action="get_user_model_values", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_user_model_values", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_user_model_values: {str(e)}")


@router.get("/model/names/{user_id}/{session_id}/")
def get_user_models(user_id: int, session_id: int, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        models = service.list_user_models(user_id=user_id, session_id=session_id, db=db)
        return models
    except HTTPException as e:
        log_error(db=db, action="get_user_models", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_user_models", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_user_models: {str(e)}")
    
@router.post("/monthly/info/")
def monthly_info(search: MonthlyInfoSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.get_monthly_info(search=search, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="monthly_info", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="monthly_info", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on monthly_info: {str(e)}")

@router.post("/day/behavior/")
def day_behavior(search: DayBehaviorSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.get_day_behavior(search=search, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="day_behavior", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="day_behavior", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on day_behavior: {str(e)}")
    
@router.post("/yearly/type/")
def create_type_year_list(search: CreateTypeYearListSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.create_type_year_list_service(search=search, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="create_type_year_list", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="create_type_year_list", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on create_type_year_list: {str(e)}")
    
@router.put("/yearly/type/")
def update_type_year_list(search: UpdateTypeYearListSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.update_type_year_list_service(search=search, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="update_type_year_list", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="update_type_year_list", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on update_type_year_list: {str(e)}")

@router.get("/yearly/type/{user_id}/{session_id}/")
def get_type_year_list(user_id: int, session_id: int, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.get_type_year_list_service(user_id=user_id, session_id=session_id, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="get_type_year_list", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_type_year_list", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_type_year_list: {str(e)}")
    
@router.post("/predict/excel/")
def predict_excel(
    search: ForecastSchema, 
    db: Session = Depends(get_db), 
    service: ForecastService = Depends(),
    current_user: str = Depends(get_current_user)
):
    try:
        excel_buffer = service.predict_excel(search=search, db=db)
        
        # Configurar headers para la descarga del archivo
        headers = {
            'Content-Disposition': 'attachment; filename="Prediccion_Caribe_Mar.xlsx"'
        }
        
        # Devolver el archivo Excel como respuesta streaming
        return StreamingResponse(
            excel_buffer,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers=headers
        )
    except HTTPException as e:
        log_error(db=db, action="predict_excel", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="predict_excel", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on predict_excel: {str(e)}")
    
@router.post("/day/behavior/excel/")
def get_day_behavior_excel(
    search: DayBehaviorSchema,
    db: Session = Depends(get_db),
    service: ForecastService = Depends(),
    current_user: str = Depends(get_current_user)
):
    try:
        print("Processing request with search parameters:", search)
        excel_buffer = service.day_behavior_excel(search=search, db=db)
        
        filename = f"Comportamiento_Diario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        
        return StreamingResponse(
            excel_buffer,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers=headers
        )
    except HTTPException as e:
        log_error(db=db, action="get_day_behavior_excel", user_id=current_user, error=e)
        print(f"HTTPException: {str(e)}")
        raise e
    except Exception as e:
        log_error(db=db, action="get_day_behavior_excel", user_id=current_user, error=e)
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Exception on day_behavior_excel: {str(e)}\nTraceback: {traceback.format_exc()}"
        )
    
@router.put("/monthly/update/")
def update_monthly_type(search: UpdateMonthlyTypeSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.update_monthly_type(search=search, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="update_monthly_type", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="update_monthly_type", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on update_monthly_type: {str(e)}")
    
@router.get("/yearly/last/", response_model=None)
def get_last_year(db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.get_last_year_on_db(db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="get_last_year", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_last_year", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_last_year: {str(e)}")
    
@router.get("/yearly/list/")
def get_yearly_list(db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.list_historic_years(db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="get_yearly_list", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_yearly_list", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_yearly_list: {str(e)}")


@router.get("/demand/monthly/{year}/")
def get_monthly_demand_by_year(year: int, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        demand = service.get_monthly_demand_by_year(db=db, year=year)
        return demand
    except HTTPException as e:
        log_error(db=db, action="get_monthly_demand_by_year", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="get_monthly_demand_by_year", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on get_monthly_demand_by_year: {str(e)}")
    
@router.post("/model/type/")
def change_model_monthly_type(search: ForecastTypeMonthSchema, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.change_model_monthly_type(search=search, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="change_model_monthly_type", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="change_model_monthly_type", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on change_model_monthly_type: {str(e)}")

@router.post("/model/savevalues/year/")
def change_model_based_on_year(search: ForecastModelSaveBasedOnYear, db: Session = Depends(get_db), service: ForecastService = Depends(), current_user: str = Depends(get_current_user)):
    try:
        info = service.change_model_based_on_year(search=search, db=db)
        return info
    except HTTPException as e:
        log_error(db=db, action="change_model_based_on_year", user_id=current_user, error=e)
        raise e
    except Exception as e:
        log_error(db=db, action="change_model_based_on_year", user_id=current_user, error=e)
        raise HTTPException(status_code=422, detail=f"Exception on change_model_based_on_year: {str(e)}")
