from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime
from fastapi import HTTPException

class RoleSchema(BaseModel):
    id: Optional[int] = None
    name: str

class LogSchema(BaseModel):
    id: Optional[int] = None
    user_id: Optional[int] = None
    action: str
    action_timestamp: Optional[datetime] = None
    details: str
    key_value: Optional[int] = None

class UserSchema(BaseModel):
    id: Optional[int] = None
    name: str
    user_name: str
    email: str
    status: int
    role_id: int
    role_name: Optional[str] = None
    created_at: Optional[datetime] = None


class ModuleSchema(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    path: str
    icono: str
    orden: int

class RolesModulesSchema(BaseModel):
    id: Optional[int] = None
    role_id: int
    module_id: int


class DemandSearchSchema(BaseModel):
    tipo: int
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None
    ano_inicio: Optional[str] = None
    ano_fin: Optional[str] = None

class DemandSchema(BaseModel):
    lista_fechas: list
    lista_demandas: list

class MacroeconomicSchema(BaseModel):
    lista_fechas: list
    lista_variables: list

class AllMacroeconomicSchema(BaseModel):
    lista_fechas: list
    lista_PIB: list
    lista_precio_oro: list
    lista_crudo_promedio: list
    lista_poblacion: list

class SearchMacroeconomicVsDemandSchema(BaseModel):
    id: int
    ano_inicio: str
    ano_fin: str

class SearchAllMacroeconomicSchema(BaseModel):
    ano_inicio: str
    ano_fin: str

class MacroeconomicVsDemandSchema(BaseModel):
    lista_fechas: list
    lista_demandas: list
    lista_variables: list

class InsertMacroeconomicSchema(BaseModel):
    id: int	
    anos: list[int]
    valores: list[float]


class DeleteMacroeconomicSchema(BaseModel):
    id: int
    anos: list[int]

class SearchAllClimateSchema(BaseModel):
    fecha_inicio: str
    fecha_fin: str

class AllClimateSchema(BaseModel):
    lista_fechas: list
    lista_temperaturas: list
    lista_humedades: list
    lista_velocidades: list

class DayClimateSchema(BaseModel):
    lista_fechas: list
    lista_variables: list   

class SearchAllClimateTypeSchema(BaseModel):
    id: int
    fecha_inicio: str
    fecha_fin: str

class SearchAllClimateTypeYearlySchema(BaseModel):
    id: int
    ano_inicio: str
    ano_fin: str

class SearchDayClimateSchema(BaseModel):
    id: int
    tipo: int
    fecha_inicio: str
    fecha_fin: str

class MonthClimateSchema(BaseModel):
    lista_fechas: list
    lista_variables: list

class SearchMonthClimateSchema(BaseModel):
    id: int
    tipo: int
    fecha_inicio: str
    fecha_fin: str

class YearClimateSchema(BaseModel):
    lista_fechas: list
    lista_variables: list

class SearchYearClimateSchema(BaseModel):
    id: int
    tipo: int
    ano_inicio: str
    ano_fin: str


class CorrelationSchema(BaseModel):
    correlation_matrix: list
    correlation_matrix_labels: list

class SearchCorrelationSchema(BaseModel):
    list_ids: Optional[list[int]] = None
    list_id_macroeconomic: Optional[list[int]] = None
    tipo: int
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None
    ano_inicio: Optional[str] = None
    ano_fin: Optional[str] = None

class ForecastSchema(BaseModel):
    id: int
    user_id: int
    session_id: int
    start_date: str
    end_date: str

class NewSessionSchema(BaseModel):
    user_id: int
    session_name: str
    description: Optional[str] = None


class NewUserModelSchema(BaseModel):
    model_name: str
    user_id: int
    session_id: int
    start_date: str
    end_date: str

class CreateOrUpdateUsersModelsSchema(BaseModel):
    model_id: int
    user_id: int
    session_id: int
    dates: list[str]
    values: list[float]


class MonthlyInfoSchema(BaseModel):
    year: int
    month: int
    predicition: float

class DayBehaviorSchema(BaseModel):
    lag: int
    start_date: str
    end_date: str
    model_type: int
    model_id: int
    user_id: Optional[int] = None
    session_id: Optional[int] = None

class CreateTypeYearListSchema(BaseModel):
    user_id: int
    session_id: int

class UpdateTypeYearListSchema(BaseModel):
    user_id: int
    session_id: int
    year: list[int]
    type: list[str]

class LoadNewAdemZipSchema(BaseModel):
    file_path: str

class UpdateMonthlyTypeSchema(BaseModel):
    values: dict 

class SearchAnalysisGridSchema(BaseModel):
    fecha_inicio: str
    fecha_fin: str


class SearchTypeDaysSchema(BaseModel):
    fecha_inicio: str
    fecha_fin: str

class MPMPredictSchema(BaseModel):
    id: int
    last_date: str
    previous_days: int
    type: int

class ForecastTypeMonthSchema(BaseModel):
    model_id: int
    dates: list[str]
    types: list[str]

class ForecastModelTypeChangeSchema(BaseModel):
    model_id: int
    dates: list[str]
    types: list[str]

class ForecastModelSaveBasedOnYear(BaseModel):
    model_id: int
    year: int
    predict_year: int

class PorcentajeCubrimientoSchema(BaseModel):
    anio: int
    valores: list[float]  # 12 valores, uno por mes

class CubrimientoRequestSchema(BaseModel):
    tipo_calculo: str  # "bolsa_neta", "venta_bolsa", "compra_bolsa"
    tipo_dato: str  # "mensual", "anual"
    tipo_grafica: str  # "area", "barras"
    mes_inicial: int
    mes_final: int
    anio: int
    porcentajes_cubrimiento: dict[str, list[float]]  # Clave: ano, Valor: lista de 12 porcentajes

class ContratoSchema(BaseModel):
    id: Optional[int] = None
    nombre: str
    fecha_creacion: Optional[datetime] = None

class PerfilSchema(BaseModel):
    id: Optional[int] = None
    nombre: str
    fecha_creacion: Optional[datetime] = None

class ResultadoCubrimientoSchema(BaseModel):
    energia_contratada: list[float]
    energia_estimada: list[float]
    desviacion: list[float]
    resultados_cubrimiento: list[float]
    energia_prevista: list[float]
    grafica: dict
    tabla: dict

# ==================== VALORACIÓN DE OFERTA ====================

class OfertaValoracionSchema(BaseModel):
    id: Optional[int] = None
    nombre: str
    fecha_carga: Optional[datetime] = None
    numero_ofertas: int
    activo: bool = True

class CargaOfertaSchema(BaseModel):
    """Schema para cargar archivo de ofertas"""
    nombre: str
    usuario_id: int

class EscenarioValoracionSchema(BaseModel):
    id: Optional[int] = None
    nombre: str
    oferta_id: int
    usuario_id: int
    fecha_creacion: Optional[datetime] = None
    ipp_base: float = 1.0  # ✅ Campo principal editable
    
    # ✅ CAMPOS CON VALORES OPTIMIZADOS DE PRODUCCIÓN - NO EDITABLES
    numero_contratos_max: Optional[int] = None  # Sin límite por defecto
    restricciones_adicionales: Optional[dict] = None
    tamano_poblacion: int = 200  # ✅ Valor optimizado para producción
    numero_generaciones: int = 100  # ✅ Valor optimizado para producción
    tolerancia_funcion: float = 1e-6  # ✅ Valor optimizado para producción
    generaciones_estancamiento: int = 25  # ✅ Valor optimizado para producción
    
    class Config:
        extra = "ignore"
        validate_assignment = True


class CrearEscenarioRequest(BaseModel):
    """Schema simplificado para crear escenario - Solo nombre e IPP"""
    nombre: str
    oferta_id: int
    usuario_id: int
    ipp_base: float = 1.0
    
    class Config:
        extra = "ignore"
        schema_extra = {
            "example": {
                "nombre": "Escenario Base 2026",
                "oferta_id": 1,
                "usuario_id": 1,
                "ipp_base": 1.0
            }
        }


class EjecutarOptimizacionSchema(BaseModel):
    """Schema para ejecutar optimización con configuración fija"""
    escenario_id: int
    tipo_optimizacion: str = "completa"  # "simple", "multiobjetivo", "completa"
    usar_poblacion_inicial: bool = True

    class Config:
        schema_extra = {
            "example": {
                "escenario_id": 1,
                "tipo_optimizacion": "completa",
                "usar_poblacion_inicial": True
            }
        }

class ResultadoOptimizacionSchema(BaseModel):
    id: Optional[int] = None
    escenario_id: int
    fecha_ejecucion: Optional[datetime] = None
    tipo_optimizacion: str
    iteracion: Optional[int] = None
    
    # ✅ CAMBIO PRINCIPAL: Matriz de porcentajes por período
    porcentajes_por_periodo: list[list[float]]  # Matriz (contratos × períodos)
    
    # ✅ MANTENER COMPATIBILIDAD: Campo para promedio (opcional para compatibilidad hacia atrás)
    porcentajes_contratos: Optional[list[float]] = None
    
    # Métricas principales
    energia_total_comprada: float
    cobertura_porcentual: float
    error_cuadratico_medio: float
    tarifa_ponderada: float
    costo_total: float
    
    # Arrays por período
    energia_mensual: list[float]
    costo_mensual: list[float]
    cobertura_mensual: list[float]
    
    # ✅ MATRICES DETALLADAS (ya existían pero las enfatizo)
    energia_por_contrato_periodo: Optional[list[list[float]]] = None
    costo_por_contrato_periodo: Optional[list[list[float]]] = None
    tipos_contratos_utilizados: Optional[list[int]] = None
    
    # Información adicional para UI
    contratos_seleccionados: Optional[list[dict]] = None
    distribucion_por_tipo: Optional[dict] = None

    class Config:
        schema_extra = {
            "example": {
                "porcentajes_por_periodo": [
                    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],  # Contrato 1 - Tipo 1
                    [75.5, 75.5, 75.5, 75.5, 75.5, 75.5, 75.5, 75.5, 75.5, 75.5, 75.5, 75.5],              # Contrato 2 - Tipo 2
                    [80.0, 85.0, 90.0, 70.0, 60.0, 65.0, 75.0, 80.0, 85.0, 90.0, 85.0, 80.0],              # Contrato 3 - Tipo 3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                           # Contrato 4 - No seleccionado
                ],
                "porcentajes_contratos": [100.0, 75.5, 77.5, 0.0],  # Promedios para compatibilidad
                "energia_por_contrato_periodo": [
                    [120.5, 115.3, 130.2, 125.8, 140.1, 135.7, 128.9, 132.4, 127.6, 133.8, 129.2, 131.5],
                    [95.2, 98.7, 102.1, 88.9, 92.4, 96.8, 100.3, 103.7, 97.5, 101.2, 94.8, 99.1],
                    [87.3, 92.8, 98.4, 76.4, 65.5, 71.0, 82.0, 87.4, 92.9, 98.5, 93.0, 87.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ]
            }
        }

class DetalleContratoSeleccionadoSchema(BaseModel):
    indice: int
    tipo: int  # 1, 2, o 3
    porcentajes_por_periodo: list[float]  # 12 valores
    energia_por_periodo: list[float]
    costo_por_periodo: list[float]
    energia_total_contrato: float
    costo_total_contrato: float


class ComparacionEscenariosSchema(BaseModel):
    """Schema para comparar múltiples escenarios"""
    escenario_ids: list[int]
    incluir_graficas: bool = True
    incluir_detalles: bool = False

class ActualizarIPPSchema(BaseModel):
    """Schema para actualizar IPP en escenario"""
    escenario_id: int
    nuevo_ipp: float

class ConfiguracionIPPSchema(BaseModel):
    id: Optional[int] = None
    fecha_vigencia: datetime
    valor_ipp: float
    descripcion: Optional[str] = None
    usuario_id: int

class ExportarResultadosSchema(BaseModel):
    escenario_ids: list[int]
    formato: str = "excel"  # "excel", "csv", "json"
    incluir_graficas: bool = True          # ✅ NUEVO
    incluir_detalles_contratos: bool = True # ✅ NUEVO
    formato_profesional: bool = True        # ✅ NUEVO

class MetricasOptimizacionSchema(BaseModel):
    """Schema para métricas detalladas de optimización"""
    cobertura_total: float
    cobertura_mensual: list[float]
    desviacion_mensual: list[float]
    tarifa_promedio: float
    costo_total: float
    costo_mensual: list[float]
    contratos_seleccionados: int
    eficiencia_economica: float  # costo/energia

class DetalleContratoSchema(BaseModel):
    """Schema para detalle de contrato en resultado"""
    indice: int
    porcentaje_seleccionado: float
    energia_total: float
    costo_total: float
    tarifa_promedio: float
    tipo_oferta: str

class ResumenEscenarioSchema(BaseModel):
    """Schema simplificado para mostrar escenarios"""
    id: int
    nombre: str
    oferta_id: int
    fecha_creacion: str
    ipp_base: float
    num_resultados: int
    estado: str  # "pendiente", "completado", "error"
    oferta_nombre: Optional[str] = None
    
    # ✅ Información técnica visible pero no prominente
    configuracion_optimizada: bool = True  # Siempre True
    tiempo_estimado: str = "2-5 minutos"  # Fijo
    poblacion: int = 500  # Fijo
    generaciones: int = 200  # Fijo



from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional


class CronConfigSchema(BaseModel):
    """Schema para configurar el cron job"""
    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    minute: int = Field(..., ge=0, le=59, description="Minuto de la hora (0-59)")
    
    @validator('hour')
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('Hour must be between 0 and 23')
        return v
    
    @validator('minute')
    def validate_minute(cls, v):
        if not 0 <= v <= 59:
            raise ValueError('Minute must be between 0 and 59')
        return v


class CronStatusSchema(BaseModel):
    """Schema para respuesta del estado del cron job"""
    id: int
    hour: int
    minute: int
    is_active: bool
    last_execution: Optional[datetime]
    last_status: str
    last_error_message: Optional[str]
    schedule_time: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CronExecutionResultSchema(BaseModel):
    """Schema para el resultado de la ejecución del cron"""
    success: bool
    execution_time: datetime
    message: str
    months_processed: Optional[list] = None
    years_updated: Optional[list] = None
    files_updated: Optional[int] = None
    error_details: Optional[str] = None

class ConvocatoriaSchema(BaseModel):
    """Schema para las convocatorias"""
    nombre: str
    mercado: str
    fecha_audiencia: datetime
    

class ContratoConvocatoriaSchema(BaseModel):
    """Schema para los contratos"""
    CODIGO: int
    CONVOCATORIA_ID: int
    DESCRIPCION: Optional[str] = None
    FECHA_INICIO: datetime
    FECHA_FIN: datetime

class ContratoConvocatoriaActualizarAgentSchema(BaseModel):
    CODIGO: int
    SIC: Optional[int] = None
    AGENTE: Optional[str] = None
    RAZON_S: Optional[str] = None
    NIT: Optional[int] = None
    DIRECCION: Optional[str] = None
    CIUDAD: Optional[str] = None
    TELEFONO: Optional[int] = None
    CODIGO_TERCERO: Optional[int] = None

class ArchivoPreparacionConvocSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    FECHA_INICIO: datetime
    CARPETA: str
    SECUENCIA_ID: Optional[int] = None

class ActualizarArchivoPreparacionConvocatoriaSchema(BaseModel):
    ID: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    FECHA_INICIO: datetime
    ACCION: str
    FECHA_CARGA: datetime
    CARPETA: str

class ArchivoPliegoConsultaSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    FECHA_INICIO: datetime
    TIPO: str
    CARPETA: str

class ActualizarArchivoPliegoConsultaSchema(BaseModel):
    ID: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    FECHA_CARGA: datetime
    FECHA_INICIO: datetime
    ACCION: str
    CARPETA: str

class ArchivoPreguntasRecibidasSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str

class ArchivoPliegosDefinitivosSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str
    FECHA_INICIO: datetime

class ArchivoOfertaReservaSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str
    FECHA_INICIO: datetime

class ArchivoRequisitosHabilitantesSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str

class ArchivoPublicacionInfoAudienciaSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str

class ArchivoOfertasCantidadesPreciosSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str
    FECHA_INICIO: datetime

class ArchivoValoracionAdjudicacionSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str

class ArchivoAudienciaPublicaSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str
    FECHA_INICIO: datetime

class ArchivoContratosFirmadosSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str
    FECHA_INICIO: datetime

class ArchivoRegistroAsicSchema(BaseModel):
    CONTRATO_CODIGO: int
    NOMBRE_ARCHIVO: str
    TIPO: str
    PLANTILLA: str
    CODIGO_SIC: int
    FECHA_INICIO: datetime
    CARPETA: str

class ActualizarArchivoRegistroAsicSchema(BaseModel):
    ID: int
    NOMBRE_ARCHIVO: str
    ACCION: str
    CARPETA: str

class ArchivoReservasPresupuestalesSchema(BaseModel):
    CONTRATO_CODIGO: str
    NOMBRE_ARCHIVO: str
    TIPO: str
    CARPETA: str

class AuditLogQueryParams(BaseModel):
    start_date: datetime
    end_date: datetime
    user_id: Optional[int] = None
    key_value: Optional[int] = None

    # Validación personalizada para asegurarse de que end_date >= start_date
    @classmethod
    def validate_dates(cls, start_date: datetime, end_date: datetime):
        if end_date < start_date:
            raise HTTPException(status_code=400, detail="La fecha de fin debe ser mayor o igual a la fecha de inicio.")
        return start_date, end_date