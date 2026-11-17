"""División de mercado (fecha límite + porcentaje) para las exportaciones.

La configuración vive en la tabla `demand_split_config` de la base de cada empresa; la crea y la
edita el backend Node. Aquí solo se LEE: si la tabla no existe o está vacía, no hay división (100 %).
Misma lógica que `DemandConfigService` del original (GMR).
"""
import calendar
from datetime import date

from sqlalchemy import text
from sqlalchemy.orm import Session


class DemandSplitConfig:
    def __init__(self, fecha_limite: date, porcentaje: float):
        self.fecha_limite = fecha_limite
        self.porcentaje = porcentaje


class DemandConfigService:
    def get_config(self, db: Session) -> DemandSplitConfig:
        try:
            # SAVEPOINT: si la tabla no existe, no se aborta la transacción de la sesión.
            with db.begin_nested():
                row = db.execute(
                    text("SELECT fecha_limite, porcentaje FROM demand_split_config ORDER BY id ASC LIMIT 1")
                ).first()
            if row and row[0] is not None and row[1] is not None:
                return DemandSplitConfig(row[0], float(row[1]))
        except Exception as e:  # tabla inexistente u otro error: sin división
            print(f"[demand_split] sin configuración de división de mercado ({type(e).__name__}); se usa 100 %")
        return DemandSplitConfig(date.today(), 100.0)


def adjust_period_value(value, year, month, config: DemandSplitConfig):
    """Ajusta un valor de MonthlyDemand (con month) o YearlyDemand (month=None) comparando el FIN del
    periodo contra fecha_limite."""
    if value is None or year is None:
        return value
    if month is None:
        period_end = date(int(year), 12, 31)
    else:
        last_day = calendar.monthrange(int(year), int(month))[1]
        period_end = date(int(year), int(month), last_day)
    if period_end <= config.fecha_limite:
        return value * ((config.porcentaje or 0) / 100.0)
    return value
