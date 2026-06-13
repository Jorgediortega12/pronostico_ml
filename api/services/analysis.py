from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import func
from ..other_models.demand import Demand
from ..other_models.economics import Macroeconomics, MacroeconomicsData
from ..other_models.climate import Climate
from ..config.schemas import DemandSearchSchema, SearchMacroeconomicVsDemandSchema, SearchAllMacroeconomicSchema, SearchAllClimateSchema
from ..config.schemas import SearchDayClimateSchema, DayClimateSchema, InsertMacroeconomicSchema, DeleteMacroeconomicSchema, MonthClimateSchema, SearchMonthClimateSchema, SearchAnalysisGridSchema
from ..config.schemas import YearClimateSchema, SearchYearClimateSchema, SearchCorrelationSchema, SearchAllClimateTypeSchema, SearchAllClimateTypeYearlySchema, SearchTypeDaysSchema
from typing import List
from statistics import mean
import pandas as pd
from datetime import datetime, timedelta
import calendar
from holidays_co import get_colombia_holidays_by_year
import psycopg2
import pandas as pd
from datetime import datetime
from statistics import mean
import os
from typing import List, Dict, Any
from sqlalchemy.orm import Session


def verify_id(db: Session, id: int):
    list_id = db.query(Macroeconomics.id).all()
    list_id = [id[0] for id in list_id]
    if id not in list_id:
        raise Exception("Id no valido")


class DemandService:
    def get_24_demands(self, db: Session, search: DemandSearchSchema) -> List[Demand]:
        if search.tipo == 0:
            demands = db.query(Demand).filter(
                Demand.fecha >= datetime.strptime(search.fecha_inicio, '%Y-%m-%d'),
                Demand.fecha <= datetime.strptime(search.fecha_fin, '%Y-%m-%d')
            ).order_by(Demand.fecha.asc()).all()
            return {
                "lista_fechas": [demand.fecha for demand in demands],
                "lista_demandas": [ [
                    demand.hora_1,
                    demand.hora_2,
                    demand.hora_3,
                    demand.hora_4,
                    demand.hora_5,
                    demand.hora_6,
                    demand.hora_7,
                    demand.hora_8,
                    demand.hora_9,
                    demand.hora_10,
                    demand.hora_11,
                    demand.hora_12,
                    demand.hora_13,
                    demand.hora_14,
                    demand.hora_15,
                    demand.hora_16,
                    demand.hora_17,
                    demand.hora_18,
                    demand.hora_19,
                    demand.hora_20,
                    demand.hora_21,
                    demand.hora_22,
                    demand.hora_23,
                    demand.hora_24
                ] for demand in demands]
            }

    def get_demands(self, db: Session, search: DemandSearchSchema) -> List[Demand]:
        if search.tipo == 0:
            demands = db.query(Demand).filter(
                Demand.fecha >= datetime.strptime(search.fecha_inicio, '%Y-%m-%d'),
                Demand.fecha <= datetime.strptime(search.fecha_fin, '%Y-%m-%d'),
                Demand.file_type == 'txf'
            ).order_by(Demand.fecha.asc()).all()
            return {
                "lista_fechas": [demand.fecha for demand in demands],
                "lista_demandas": [ [
                    demand.hora_1,
                    demand.hora_2,
                    demand.hora_3,
                    demand.hora_4,
                    demand.hora_5,
                    demand.hora_6,
                    demand.hora_7,
                    demand.hora_8,
                    demand.hora_9,
                    demand.hora_10,
                    demand.hora_11,
                    demand.hora_12,
                    demand.hora_13,
                    demand.hora_14,
                    demand.hora_15,
                    demand.hora_16,
                    demand.hora_17,
                    demand.hora_18,
                    demand.hora_19,
                    demand.hora_20,
                    demand.hora_21,
                    demand.hora_22,
                    demand.hora_23,
                    demand.hora_24,
                    demand.total
                ] for demand in demands]
            }
        
        elif search.tipo == 1:
            demands = db.query(Demand).filter(
                Demand.fecha >= datetime.strptime(search.fecha_inicio, '%Y-%m-%d'),
                Demand.fecha <= datetime.strptime(search.fecha_fin, '%Y-%m-%d'),
                Demand.file_type == 'txf'
            ).order_by(Demand.fecha.asc()).all()
            return {
                "lista_fechas": [demand.fecha.strftime('%Y-%m-%d') for demand in demands],
                "lista_demandas": [ demand.total for demand in demands]
            }
        
        elif search.tipo == 2:
            demands = db.query(
                func.extract('YEAR', Demand.fecha).label('year'),
                func.extract('MONTH', Demand.fecha).label('month'),
                func.sum(Demand.total).label('total')
            ).filter(
                Demand.fecha >= datetime.strptime(search.fecha_inicio, '%Y-%m-%d'),
                Demand.fecha <= datetime.strptime(search.fecha_fin, '%Y-%m-%d'),
                Demand.file_type == 'txf'
            ).group_by(
                func.extract('YEAR', Demand.fecha),
                func.extract('MONTH', Demand.fecha)
            ).all()

            lista_fechas = [f"{int(demand.year)}-{int(demand.month):02d}" for demand in demands]
            lista_demandas = [demand.total for demand in demands]	
            year_month = [fecha for fecha in lista_fechas]
            set_year_month = set(year_month)
            sorted_set_year_month = sorted(set_year_month)
            sorted_demands = []

            for year in sorted_set_year_month:
                sorted_demands.append(sum([lista_demandas[i] for i in range(len(year_month)) if year_month[i] == year]))

            return {
                "lista_fechas": sorted_set_year_month,
                "lista_demandas": sorted_demands
            }
        
        elif search.tipo == 3:
            ano_inicio = int(search.ano_inicio)
            ano_fin = int(search.ano_fin)
            query = db.query(
                func.extract('YEAR', Demand.fecha).label('year'),
                func.sum(Demand.total).label('total')
            ).filter(
                Demand.fecha >= datetime.strptime(f"{ano_inicio}-01-01", '%Y-%m-%d'),
                Demand.fecha <= datetime.strptime(f"{ano_fin}-12-31", '%Y-%m-%d'),
                Demand.file_type == 'txf'
            ).group_by(
                func.extract('YEAR', Demand.fecha)
            )

            results = query.all()
            lista_fechas = [result.year for result in results]
            lista_demandas = [result.total for result in results]
            sorted_demands = sorted(zip(lista_fechas, lista_demandas))
            lista_fechas, lista_demandas = zip(*sorted_demands)
            lista_demandas = [lista_demandas[lista_fechas.index(fecha)] for fecha in lista_fechas]
            return {
                "lista_fechas": lista_fechas,
                "lista_demandas": lista_demandas
            }
        
        elif search.tipo == 4:
            demands = db.query(Demand).filter(
                Demand.fecha >= datetime.strptime(search.fecha_inicio, '%Y-%m-%d'),
                Demand.fecha <= datetime.strptime(search.fecha_fin, '%Y-%m-%d')
            ).order_by(Demand.fecha.asc()).all()
            lista_fechas_con_hora = [f"{demand.fecha.strftime('%Y-%m-%d')} {i}" for demand in demands for i in range(1, 25)]
            lista_demandas = [hora for demand in demands for hora in [
                demand.hora_1,
                demand.hora_2,
                demand.hora_3,
                demand.hora_4,
                demand.hora_5,
                demand.hora_6,
                demand.hora_7,
                demand.hora_8,
                demand.hora_9,
                demand.hora_10,
                demand.hora_11,
                demand.hora_12,
                demand.hora_13,
                demand.hora_14,
                demand.hora_15,
                demand.hora_16,
                demand.hora_17,
                demand.hora_18,
                demand.hora_19,
                demand.hora_20,
                demand.hora_21,
                demand.hora_22,
                demand.hora_23,
                demand.hora_24
            ]]
            return {
                "lista_fechas": lista_fechas_con_hora,
                "lista_demandas": lista_demandas
            }
        

        else:
            raise Exception("Tipo no valido")
        
    def get_first_and_last_date(self, db: Session):
        first_date = db.query(Demand.fecha).order_by(Demand.fecha).first()
        last_date = db.query(
            Demand.fecha
        ).filter(
            func.extract('day', Demand.fecha) == 31,
            func.extract('month', Demand.fecha) == 12
        ).order_by(
            Demand.fecha.desc()
        ).first()
        return {
            "first_date": first_date[0],
            "last_date": last_date[0]
        }

    def get_first_and_last_demand_date(self, db: Session):
        first_date = db.query(Demand.fecha).filter(Demand.file_type == 'txf').order_by(Demand.fecha).first()
        last_date = db.query(Demand.fecha).filter(Demand.file_type == 'txf').order_by(Demand.fecha.desc()).first()
        if not first_date or not last_date:
            raise Exception("No hay datos de demanda en la base de datos")
        return {
            "first_date": first_date[0].strftime('%Y-%m-%d'),
            "last_date": last_date[0].strftime('%Y-%m-%d')
        }
class MacroeconomicService:
    def get_all_variables(self, db: Session, search: SearchAllMacroeconomicSchema):
        """
        Obtiene todas las variables macroeconómicas para el rango de anos especificado.
        """
        # Obtener variables para el rango de anos
        variables = db.query(MacroeconomicsData).filter(
            MacroeconomicsData.ano >= search.ano_inicio,
            MacroeconomicsData.ano <= search.ano_fin
        ).order_by(MacroeconomicsData.eco_id, MacroeconomicsData.ano).all()
        
        # Verificar si hay variables disponibles
        if not variables:
            print(f"No se encontraron variables macroeconómicas para el rango: {search.ano_inicio} - {search.ano_fin}")
            return {}  # Retornar diccionario vacío si no hay datos
        
        # Obtener los IDs únicos de las variables macroeconómicas
        list_ids = [variable.eco_id for variable in variables]
        list_ids = list(set(list_ids))
        
        # Si no hay IDs, retornar diccionario vacío
        if not list_ids:
            print("No se encontraron IDs de variables macroeconómicas")
            return {}
        
        # Obtener los nombres de las variables
        list_names = [name[0] for name in db.query(Macroeconomics.name).filter(Macroeconomics.id.in_(list_ids)).all()]
        
        # Verificar que tengamos nombres para cada ID
        if not list_names or len(list_names) != len(list_ids):
            print(f"Error: No coincide el número de nombres ({len(list_names)}) con el número de IDs ({len(list_ids)})")
            return {}
        
        # Ordenar las variables
        sorted_macro = sorted(zip(list_ids, list_names))
        
        # Verificar que sorted_macro no esté vacío antes de desempaquetar
        if not sorted_macro:
            print("Error: No hay variables para ordenar")
            return {}
        
        # Desempaquetar de manera segura
        list_ids, list_names = zip(*sorted_macro)
        
        # Obtener las fechas únicas y ordenarlas
        list_dates = [variable.ano for variable in variables]
        list_dates = sorted(set(list_dates))
        
        # Construir la respuesta
        response = {
            "lista_fechas": list_dates
        }
        
        # Agregamos control para evitar índices fuera de rango
        for i, id in enumerate(list_ids):
            if i < len(list_names):
                response[list_names[i]] = [variable.value for variable in variables if variable.eco_id == id]
            else:
                print(f"Advertencia: No hay nombre para el ID {id}")
        
        return response

    def get_variable(self, db: Session, id: int):
        verify_id(db=db, id=id)
        variable = db.query(MacroeconomicsData).filter(MacroeconomicsData.eco_id == id).order_by(MacroeconomicsData.ano).all()
        if not variable:
            raise Exception("Id no valido o no hay datos para el id seleccionado")
        list_dates = [variable.ano for variable in variable]
        list_dates = sorted(set(list_dates))
        response = {
            "lista_fechas": list_dates,
            "lista_variables": [variable.value for variable in variable]
        }
        return response
    
    def get_variable_year(self, db: Session, id: int, fecha_inicio: str, fecha_fin: str):
        verify_id(db=db, id=id)
        variable = db.query(MacroeconomicsData).filter(MacroeconomicsData.eco_id == id, MacroeconomicsData.ano >= fecha_inicio, MacroeconomicsData.ano <= fecha_fin).order_by(MacroeconomicsData.ano).all()
        if not variable:
            raise Exception("Id no valido o no hay datos para el id seleccionado")
        list_dates = [variable.ano for variable in variable]
        list_dates = sorted(set(list_dates))
        response = {
            "lista_fechas": list_dates,
            "lista_variables": [variable.value for variable in variable]
        }
        return response
        
    def variable_vs_demand(self, db: Session, search: SearchMacroeconomicVsDemandSchema):
        verify_id(db=db, id=search.id)
        demand_service = DemandService()
        demand_data = demand_service.get_demands(db=db, search=
        DemandSearchSchema(
            tipo=3,
            ano_inicio=search.ano_inicio,
            ano_fin=search.ano_fin
        ))
        if not demand_data:
            raise Exception("No hay datos de demanda para el rango de fechas seleccionado")

        variable_data = db.query(
            MacroeconomicsData
        ).filter(
            MacroeconomicsData.ano >= search.ano_inicio,
            MacroeconomicsData.ano <= search.ano_fin,
            MacroeconomicsData.eco_id == search.id
        ).order_by(
            MacroeconomicsData.ano
        ).all()

        if not variable_data:
            raise Exception("No hay datos de la variable seleccionada para el rango de fechas seleccionado")

        return {
            "lista_fechas": demand_data["lista_fechas"],
            "lista_demandas": demand_data["lista_demandas"],
            "lista_variables": [variable.value for variable in variable_data]
        }

    def insert_or_update_variable(self, db: Session, search: InsertMacroeconomicSchema):
        verify_id(db=db, id=search.id)
        current = db.query(MacroeconomicsData.value, MacroeconomicsData.ano).filter(MacroeconomicsData.ano.in_(search.anos), MacroeconomicsData.eco_id == search.id).all()
        if current:
            anos_not_created = [ano for ano in search.anos if ano not in [c[1] for c in current]]
            for ano in anos_not_created:
                db.add(MacroeconomicsData(
                    value=search.valores[search.anos.index(ano)],
                    ano=ano,
                    eco_id=search.id
                ))
            anos_to_update = [ano for ano in search.anos if ano in [c[1] for c in current]]
            for ano in anos_to_update:
                db.query(MacroeconomicsData).filter(MacroeconomicsData.ano == ano, MacroeconomicsData.eco_id == search.id).update({"value": search.valores[search.anos.index(ano)]})

        else:
            for i in range(len(search.anos)):
                db.add(MacroeconomicsData(
                    value=search.valores[i],
                    ano=search.anos[i],
                    eco_id=search.id
                ))
        
        db.commit()
        return {
            "valido": "Valores insertados o actualizados correctamente"
        }

    def delete_column_values(self, db: Session, search: DeleteMacroeconomicSchema): 
        verify_id(db=db, id=search.id)
        db.query(MacroeconomicsData).filter(MacroeconomicsData.eco_id == search.id, MacroeconomicsData.ano.in_(search.anos)).delete(synchronize_session=False)
        db.commit()

        return {
            "valido": "Valores eliminados correctamente"
        }
    

    def get_economics_ids(self, db: Session):
        ids_and_names = db.query(Macroeconomics.id, Macroeconomics.name).all()
        json = {}
        macroeconomic_variables = {
            'ipc': 'ipc',
            'Crecimiento': 'Crecimiento de la población (%) anual',
            'Inflación': 'Inflación, deflactor del PIB: series vinculadas (% anual)',
            'Comercio': 'Comercio de mercaderías (% del PIB)',
            'Importaciones': 'Importaciones de bienes (balanza de pagos, US$ a precios actuales)',
            'Empleo': 'Empleo de tiempo parcial, total (% del total de empleo)'
        }
        for id, name in ids_and_names:
            json[id] = macroeconomic_variables[name]
        return json  

    def get_first_and_last_date_of_each_variable(self, db: Session):
        try:	
            first_dates = db.query(
                MacroeconomicsData.eco_id,
                func.min(MacroeconomicsData.ano).label('first_date')
            ).group_by(MacroeconomicsData.eco_id).all()

            last_dates = db.query(
                MacroeconomicsData.eco_id,
                func.max(MacroeconomicsData.ano).label('last_date')
            ).group_by(MacroeconomicsData.eco_id).all()


            # Get all macroeconomic variable names
            variables = db.query(Macroeconomics.id, Macroeconomics.name).all()
            var_names_dict = {var.id: var.name for var in variables}

            first_last_dates = {}
            for first in first_dates:
                for last in last_dates:
                    if first.eco_id == last.eco_id:
                        # Get the name of the variable instead of using the ID
                        var_name = var_names_dict.get(first.eco_id, f"Unknown Variable {first.eco_id}")
                        first_last_dates[var_name] = {
                            "first_date": first.first_date,
                            "last_date": last.last_date
                        }
            return first_last_dates   
        except SQLAlchemyError as e:
            print(f"Error al obtener las fechas: {e}")
            raise Exception("Error al obtener las fechas de las variables macroeconómicas")

import psycopg2
import pandas as pd
from datetime import datetime
from statistics import mean
import os
from typing import List, Dict, Any
from sqlalchemy.orm import Session

class ClimateService:
    def __init__(self):
        """Inicializa el servicio con las credenciales de PostgreSQL desde variables de entorno"""
        self.host = os.getenv("POSTGRES_HOST2")
        self.port = os.getenv("POSTGRES_PORT2", 5435)
        self.database = os.getenv("POSTGRES_DATABASE2")
        self.user = os.getenv("POSTGRES_USER2")
        self.password = os.getenv("POSTGRES_PASSWORD2")
    
    def _get_connection(self):
        """Establece conexión con PostgreSQL"""
        try:
            connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            print("conexion:", connection)
            return connection
        except psycopg2.Error as e:
            raise Exception(f"Error al conectar a PostgreSQL: {e}")
    
    def _extract_hourly_data(self, df: pd.DataFrame, data_type: str) -> List[float]:
        """Extrae datos horarios de un DataFrame basado en el tipo de dato"""
        hourly_data = []
        prefix_map = {
            'temperatura': 'p{}_t',
            'humedad': 'p{}_h', 
            'velocidad': 'p{}_v'
        }
        
        if data_type not in prefix_map:
            raise ValueError(f"Tipo de dato no válido: {data_type}")
        
        prefix = prefix_map[data_type]
        
        for _, row in df.iterrows():
            day_data = []
            for hour in range(1, 25):
                col_name = prefix.format(hour)
                if col_name in row and pd.notna(row[col_name]):
                    day_data.append(float(row[col_name]))
                else:
                    raise ValueError(f"Datos faltantes para {data_type} en hora {hour}")
            hourly_data.extend(day_data)
        
        return hourly_data

    def get_climate_hour(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos por hora para un rango de fechas
        """
        connection = self._get_connection()
        
        try:
            # Query para obtener datos climáticos
            query = """
            SELECT fecha, p1_t, p2_t, p3_t, p4_t, p5_t, p6_t, p7_t, p8_t, p9_t, p10_t,
                   p11_t, p12_t, p13_t, p14_t, p15_t, p16_t, p17_t, p18_t, p19_t, p20_t,
                   p21_t, p22_t, p23_t, p24_t,
                   p1_h, p2_h, p3_h, p4_h, p5_h, p6_h, p7_h, p8_h, p9_h, p10_h,
                   p11_h, p12_h, p13_h, p14_h, p15_h, p16_h, p17_h, p18_h, p19_h, p20_h,
                   p21_h, p22_h, p23_h, p24_h,
                   p1_v, p2_v, p3_v, p4_v, p5_v, p6_v, p7_v, p8_v, p9_v, p10_v,
                   p11_v, p12_v, p13_v, p14_v, p15_v, p16_v, p17_v, p18_v, p19_v, p20_v,
                   p21_v, p22_v, p23_v, p24_v
            FROM datos_clima 
            WHERE fecha >= %s AND fecha <= %s
            ORDER BY fecha
            """
            
            df = pd.read_sql_query(
                query, 
                connection, 
                params=[search.fecha_inicio, search.fecha_fin]
            )
            
            if df.empty:
                raise ValueError(f"No se encontraron datos climáticos para el rango {search.fecha_inicio} - {search.fecha_fin}")
            
            # Generar lista de fechas con horas
            lista_fechas_con_hora = []
            for _, row in df.iterrows():
                fecha_str = row['fecha'].strftime('%Y-%m-%d')
                for hour in range(1, 25):
                    lista_fechas_con_hora.append(f"{fecha_str} {hour}")
            
            # Extraer datos horarios para cada tipo
            lista_temperaturas = self._extract_hourly_data(df, 'temperatura')
            lista_humedades = self._extract_hourly_data(df, 'humedad')
            lista_velocidades = self._extract_hourly_data(df, 'velocidad')
            
            return {
                "lista_fechas": lista_fechas_con_hora,
                "lista_temperaturas": lista_temperaturas,
                "lista_humedades": lista_humedades,
                "lista_velocidades": lista_velocidades
            }
            
        finally:
            connection.close()

    def get_climate(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos agrupados por día
        """
        connection = self._get_connection()
        
        try:
            # MODIFICADO: Agregado filtro por UCP = 'Barranquilla'
            query = """
            SELECT fecha, p1_t, p2_t, p3_t, p4_t, p5_t, p6_t, p7_t, p8_t, p9_t, p10_t,
                p11_t, p12_t, p13_t, p14_t, p15_t, p16_t, p17_t, p18_t, p19_t, p20_t,
                p21_t, p22_t, p23_t, p24_t,
                p1_h, p2_h, p3_h, p4_h, p5_h, p6_h, p7_h, p8_h, p9_h, p10_h,
                p11_h, p12_h, p13_h, p14_h, p15_h, p16_h, p17_h, p18_h, p19_h, p20_h,
                p21_h, p22_h, p23_h, p24_h,
                p1_v, p2_v, p3_v, p4_v, p5_v, p6_v, p7_v, p8_v, p9_v, p10_v,
                p11_v, p12_v, p13_v, p14_v, p15_v, p16_v, p17_v, p18_v, p19_v, p20_v,
                p21_v, p22_v, p23_v, p24_v
            FROM datos_clima 
            WHERE fecha >= %s AND fecha <= %s AND ucp = %s
            ORDER BY fecha
            """
            
            # MODIFICADO: Agregado 'Barranquilla' como tercer parámetro
            df = pd.read_sql_query(
                query, 
                connection, 
                params=[search.fecha_inicio, search.fecha_fin, 'Bolivar']
            )
            
            if df.empty:
                raise ValueError(f"No se encontraron datos climáticos para Barranquilla en el rango {search.fecha_inicio} - {search.fecha_fin}")
            
            lista_fechas = [row['fecha'].strftime('%Y-%m-%d') for _, row in df.iterrows()]
            
            # Extraer datos por día para cada tipo
            lista_temperaturas = []
            lista_humedades = []
            lista_velocidades = []
            
            for _, row in df.iterrows():
                # Temperaturas del día
                temp_day = []
                for hour in range(1, 25):
                    col_name = f"p{hour}_t"
                    if col_name in row and pd.notna(row[col_name]):
                        temp_day.append(float(row[col_name]))
                    else:
                        temp_day.append(0.0)
                lista_temperaturas.append(temp_day)
                
                # Humedades del día
                hum_day = []
                for hour in range(1, 25):
                    col_name = f"p{hour}_h"
                    if col_name in row and pd.notna(row[col_name]):
                        hum_day.append(float(row[col_name]))
                    else:
                        hum_day.append(0.0)
                lista_humedades.append(hum_day)
                
                # Velocidades del día
                vel_day = []
                for hour in range(1, 25):
                    col_name = f"p{hour}_v"
                    if col_name in row and pd.notna(row[col_name]):
                        vel_day.append(float(row[col_name]))
                    else:
                        vel_day.append(0.0)
                lista_velocidades.append(vel_day)
                
            return {
                "lista_fechas": lista_fechas,
                "lista_temperaturas": lista_temperaturas,
                "lista_humedades": lista_humedades,
                "lista_velocidades": lista_velocidades
            }
            
        finally:
            connection.close()
            
    def get_climate_day(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos diarios con agregación (promedio, máximo, mínimo)
        """
        # Crear un objeto search compatible para get_climate
        class SearchAllClimateSchema:
            def __init__(self, fecha_inicio, fecha_fin):
                self.fecha_inicio = fecha_inicio
                self.fecha_fin = fecha_fin
        
        sub_search = SearchAllClimateSchema(
            fecha_inicio=search.fecha_inicio,
            fecha_fin=search.fecha_fin
        )
        
        variables = self.get_climate(db=db, search=sub_search)
        
        # Determinar función de agregación
        if search.tipo == 0:
            action = mean
        elif search.tipo == 1:
            action = max
        elif search.tipo == 2:
            action = min
        else:
            raise Exception("tipo no valido")

        # Determinar qué tipo de variable usar
        if search.id == 0:
            key = "lista_temperaturas"
        elif search.id == 1:
            key = "lista_humedades"
        elif search.id == 2:
            key = "lista_velocidades"
        else:
            raise Exception("id no valido")
        
        # Aplicar agregación a cada día
        values = []
        for lista in variables[key]:
            # Filtrar valores válidos (no None, no NaN)
            valid_values = [v for v in lista if v is not None and not pd.isna(v)]
            if not valid_values:
                raise ValueError(f"No hay datos válidos para el tipo {key}")
            values.append(round(action(valid_values), 2))
        
        return {
            "lista_fechas": variables["lista_fechas"],
            "lista_variables": values
        }
    
    def get_climate_month(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos mensuales con agregación
        """
        # Crear un objeto search compatible para get_climate_day
        class SearchDayClimateSchema:
            def __init__(self, id, tipo, fecha_inicio, fecha_fin):
                self.id = id
                self.tipo = tipo
                self.fecha_inicio = fecha_inicio
                self.fecha_fin = fecha_fin
        
        sub_search = SearchDayClimateSchema(
            id=search.id,
            tipo=search.tipo,
            fecha_inicio=search.fecha_inicio,
            fecha_fin=search.fecha_fin
        )
        
        variables = self.get_climate_day(db=db, search=sub_search)
        year_month = [fecha[:7] for fecha in variables["lista_fechas"]]
        set_year_month = set(year_month)
        sorted_set_year_month = sorted(set_year_month)
        values = []

        # Determinar función de agregación
        if search.tipo == 0:
            action = mean
        elif search.tipo == 1:
            action = max
        elif search.tipo == 2:
            action = min
        else:
            raise Exception("tipo no valido")

        # Agrupar por mes y aplicar agregación
        for year in sorted_set_year_month:
            month_values = [variables["lista_variables"][i] 
                          for i in range(len(year_month)) 
                          if year_month[i] == year]
            if not month_values:
                raise ValueError(f"No hay datos válidos para el mes {year}")
            values.append(round(action(month_values), 2))

        return {
            "lista_fechas": sorted_set_year_month,
            "lista_variables": values
        }
    
    def get_climate_year(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos anuales con agregación
        """
        ano_inicio = search.ano_inicio
        ano_fin = search.ano_fin
        fecha_inicio = f"{ano_inicio}-01-01"
        fecha_fin = f"{ano_fin}-12-31"

        # Crear un objeto search compatible para get_climate_month
        class SearchMonthClimateSchema:
            def __init__(self, id, tipo, fecha_inicio, fecha_fin):
                self.id = id
                self.tipo = tipo
                self.fecha_inicio = fecha_inicio
                self.fecha_fin = fecha_fin

        sub_search = SearchMonthClimateSchema(
            id=search.id,
            tipo=search.tipo,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin
        )
        
        variables = self.get_climate_month(db=db, search=sub_search)
        year = [fecha[:4] for fecha in variables["lista_fechas"]]
        set_year = set(year)
        sorted_set_year = sorted(set_year)
        values = []
        
        # Determinar función de agregación
        if search.tipo == 0:
            action = mean
        elif search.tipo == 1:
            action = max
        elif search.tipo == 2:
            action = min
        else:
            raise Exception("tipo no valido")
        
        # Agrupar por ano y aplicar agregación
        for year_item in sorted_set_year:
            year_values = [variables["lista_variables"][i] 
                          for i in range(len(year)) 
                          if year[i] == year_item]
            if not year_values:
                raise ValueError(f"No hay datos válidos para el ano {year_item}")
            values.append(round(action(year_values), 2))

        return {
            "lista_fechas": sorted_set_year,
            "lista_variables": values
        }

    def get_climate_day_all_types(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos diarios con todos los tipos de agregación (max, min, avg)
        """
        try:
            # Crear un objeto search compatible para get_climate_day
            class SearchDayClimateSchema:
                def __init__(self, fecha_inicio, fecha_fin, id, tipo):
                    self.fecha_inicio = fecha_inicio
                    self.fecha_fin = fecha_fin
                    self.id = id
                    self.tipo = tipo

            maxs = self.get_climate_day(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=search.id, 
                tipo=1
            ))
            mins = self.get_climate_day(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=search.id, 
                tipo=2
            ))
            avgs = self.get_climate_day(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=search.id, 
                tipo=0
            ))
            
            dates = maxs["lista_fechas"]
            max_values = maxs["lista_variables"]
            min_values = mins["lista_variables"]
            avg_values = avgs["lista_variables"]
            
            return {
                "lista_fechas": dates,
                "lista_max": max_values,
                "lista_min": min_values,
                "lista_avg": avg_values
            }
        except Exception as e:
            raise e
        
    def get_climate_month_all_types(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos mensuales con todos los tipos de agregación (max, min, avg)
        """
        try:
            # Crear un objeto search compatible para get_climate_month
            class SearchMonthClimateSchema:
                def __init__(self, fecha_inicio, fecha_fin, id, tipo):
                    self.fecha_inicio = fecha_inicio
                    self.fecha_fin = fecha_fin
                    self.id = id
                    self.tipo = tipo

            maxs = self.get_climate_month(db=db, search=SearchMonthClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=search.id, 
                tipo=1
            ))
            mins = self.get_climate_month(db=db, search=SearchMonthClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=search.id, 
                tipo=2
            ))
            avgs = self.get_climate_month(db=db, search=SearchMonthClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=search.id, 
                tipo=0
            ))
            
            dates = maxs["lista_fechas"]
            max_values = maxs["lista_variables"]
            min_values = mins["lista_variables"]
            avg_values = avgs["lista_variables"]
            
            return {
                "lista_fechas": dates,
                "lista_max": max_values,
                "lista_min": min_values,
                "lista_avg": avg_values
            }
        except Exception as e:
            raise e

    def get_climate_year_all_types(self, db: Session, search) -> Dict[str, Any]:
        """
        Obtiene datos climáticos anuales con todos los tipos de agregación (max, min, avg)
        """
        try:
            # Crear un objeto search compatible para get_climate_year
            class SearchYearClimateSchema:
                def __init__(self, ano_inicio, ano_fin, id, tipo):
                    self.ano_inicio = ano_inicio
                    self.ano_fin = ano_fin
                    self.id = id
                    self.tipo = tipo

            maxs = self.get_climate_year(db=db, search=SearchYearClimateSchema(
                ano_inicio=search.ano_inicio, 
                ano_fin=search.ano_fin, 
                id=search.id, 
                tipo=1
            ))
            mins = self.get_climate_year(db=db, search=SearchYearClimateSchema(
                ano_inicio=search.ano_inicio, 
                ano_fin=search.ano_fin, 
                id=search.id, 
                tipo=2
            ))
            avgs = self.get_climate_year(db=db, search=SearchYearClimateSchema(
                ano_inicio=search.ano_inicio, 
                ano_fin=search.ano_fin, 
                id=search.id, 
                tipo=0
            ))
            
            dates = maxs["lista_fechas"]
            max_values = maxs["lista_variables"]
            min_values = mins["lista_variables"]
            avg_values = avgs["lista_variables"]
            
            return {
                "lista_fechas": dates,
                "lista_max": max_values,
                "lista_min": min_values,
                "lista_avg": avg_values
            }
        except Exception as e:
            raise e
        

    def get_climate_per_city(self, db: Session, search, city) -> Dict[str, Any]:
        """
        Obtiene datos climáticos agrupados por día para una ciudad específica
        """
        connection = self._get_connection()
        
        try:
            query = """
            SELECT fecha, p1_t, p2_t, p3_t, p4_t, p5_t, p6_t, p7_t, p8_t, p9_t, p10_t,
                p11_t, p12_t, p13_t, p14_t, p15_t, p16_t, p17_t, p18_t, p19_t, p20_t,
                p21_t, p22_t, p23_t, p24_t,
                p1_h, p2_h, p3_h, p4_h, p5_h, p6_h, p7_h, p8_h, p9_h, p10_h,
                p11_h, p12_h, p13_h, p14_h, p15_h, p16_h, p17_h, p18_h, p19_h, p20_h,
                p21_h, p22_h, p23_h, p24_h,
                p1_v, p2_v, p3_v, p4_v, p5_v, p6_v, p7_v, p8_v, p9_v, p10_v,
                p11_v, p12_v, p13_v, p14_v, p15_v, p16_v, p17_v, p18_v, p19_v, p20_v,
                p21_v, p22_v, p23_v, p24_v
            FROM datos_clima 
            WHERE fecha >= %s AND fecha <= %s AND ucp = %s
            ORDER BY fecha
            """
            
            df = pd.read_sql_query(
                query, 
                connection, 
                params=[search.fecha_inicio, search.fecha_fin, city]
            )
            print(connection)

            # print(f"\n{'='*60}")
            # print(f"DATOS CLIMA - Ciudad: {city}")
            # print(f"Rango: {search.fecha_inicio} → {search.fecha_fin}")
            # print(f"Filas encontradas: {len(df)}")
            # print(f"Columnas: {list(df.columns)}")
            # if not df.empty:
            #     print(f"\nPrimera fila completa:")
            #     print(df.iloc[0].to_string())
            #     print(f"\nResumen temperatura (p1_t a p24_t):")
            #     temp_cols = [f"p{h}_t" for h in range(1, 25)]
            #     print(df[temp_cols].describe())
            # print(f"{'='*60}\n")
            
            if df.empty:
                # CORREGIDO: Usar variable city en lugar de 'Barranquilla' hardcodeado
                raise ValueError(f"No se encontraron datos climáticos para {city} en el rango {search.fecha_inicio} - {search.fecha_fin}")
            
            lista_fechas = [row['fecha'].strftime('%Y-%m-%d') for _, row in df.iterrows()]
            
            # Extraer datos por día para cada tipo
            lista_temperaturas = []
            lista_humedades = []
            lista_velocidades = []
            
            for _, row in df.iterrows():
                # Temperaturas del día (24 horas)
                temp_day = []
                for hour in range(1, 25):
                    col_name = f"p{hour}_t"
                    if col_name in row and pd.notna(row[col_name]):
                        temp_day.append(float(row[col_name]))
                    else:
                        temp_day.append(0.0)
                lista_temperaturas.append(temp_day)
                
                # Humedades del día (24 horas)
                hum_day = []
                for hour in range(1, 25):
                    col_name = f"p{hour}_h"
                    if col_name in row and pd.notna(row[col_name]):
                        hum_day.append(float(row[col_name]))
                    else:
                        hum_day.append(0.0)
                lista_humedades.append(hum_day)
                
                # Velocidades del día (24 horas)
                vel_day = []
                for hour in range(1, 25):
                    col_name = f"p{hour}_v"
                    if col_name in row and pd.notna(row[col_name]):
                        vel_day.append(float(row[col_name]))
                    else:
                        vel_day.append(0.0)
                lista_velocidades.append(vel_day)
                
            return {
                "ciudad": city,
                "lista_fechas": lista_fechas,
                "temperaturas": lista_temperaturas,
                "humedades": lista_humedades,
                "velocidades": lista_velocidades
            }    
        finally:
            connection.close()


    def get_climate_daily_per_city(self, db: Session, search, city) -> Dict[str, Any]:
        """
        Obtiene datos climáticos diarios con agregación (promedio, máximo, mínimo) para una ciudad específica
        """
        # Crear un objeto search compatible para get_climate_per_city
        class SearchAllClimateSchema:
            def __init__(self, fecha_inicio, fecha_fin):
                self.fecha_inicio = fecha_inicio
                self.fecha_fin = fecha_fin
        
        sub_search = SearchAllClimateSchema(
            fecha_inicio=search.fecha_inicio,
            fecha_fin=search.fecha_fin
        )
        
        variables = self.get_climate_per_city(db=db, search=sub_search, city=city)
        
        # Determinar función de agregación
        if search.tipo == 0:
            action = mean
        elif search.tipo == 1:
            action = max
        elif search.tipo == 2:
            action = min
        else:
            raise Exception("tipo no valido")

        # Determinar qué tipo de variable usar
        if search.id == 0:
            key = "temperaturas"
        elif search.id == 1:
            key = "humedades"
        elif search.id == 2:
            key = "velocidades"
        else:
            raise Exception("id no valido")
        
        # Aplicar agregación a cada día
        values = []
        for lista in variables[key]:
            # Filtrar valores válidos (no None, no NaN)
            valid_values = [v for v in lista if v is not None and not pd.isna(v) and v != 0.0]
            if not valid_values:
                raise ValueError(f"No hay datos válidos para el tipo {key} en {city}")
            values.append(round(action(valid_values), 2))
        
        return {
            "ciudad": city,
            "lista_fechas": variables["lista_fechas"],
            "tipo_variable": key,
            "tipo_agregacion": "promedio" if search.tipo == 0 else "maximo" if search.tipo == 1 else "minimo",
            "valores": values
        }


    def get_climate_daily_all_types_per_city(self, db: Session, search, city) -> Dict[str, Any]:
        """
        Obtiene datos climáticos diarios con TODOS los tipos de agregación para una ciudad específica
        Retorna temperaturas, humedades y velocidades con max, min, avg
        """
        try:
            # Crear un objeto search compatible
            class SearchDayClimateSchema:
                def __init__(self, fecha_inicio, fecha_fin, id, tipo):
                    self.fecha_inicio = fecha_inicio
                    self.fecha_fin = fecha_fin
                    self.id = id
                    self.tipo = tipo

            # Obtener datos para TEMPERATURAS (id=0)
            temp_maxs = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=0, tipo=1  # temperaturas, máximo
            ), city=city)
            
            temp_mins = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=0, tipo=2  # temperaturas, mínimo
            ), city=city)
            
            temp_avgs = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=0, tipo=0  # temperaturas, promedio
            ), city=city)

            # Obtener datos para HUMEDADES (id=1)
            hum_maxs = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=1, tipo=1  # humedades, máximo
            ), city=city)
            
            hum_mins = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=1, tipo=2  # humedades, mínimo
            ), city=city)
            
            hum_avgs = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=1, tipo=0  # humedades, promedio
            ), city=city)

            # Obtener datos para VELOCIDADES (id=2)
            vel_maxs = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=2, tipo=1  # velocidades, máximo
            ), city=city)
            
            vel_mins = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=2, tipo=2  # velocidades, mínimo
            ), city=city)
            
            vel_avgs = self.get_climate_daily_per_city(db=db, search=SearchDayClimateSchema(
                fecha_inicio=search.fecha_inicio, 
                fecha_fin=search.fecha_fin, 
                id=2, tipo=0  # velocidades, promedio
            ), city=city)
            
            return {
                "ciudad": city,
                "fechas": temp_maxs["lista_fechas"],
                "temperaturas": {
                    "maximas": temp_maxs["valores"],
                    "minimas": temp_mins["valores"],
                    "promedios": temp_avgs["valores"]
                },
                "humedades": {
                    "maximas": hum_maxs["valores"],
                    "minimas": hum_mins["valores"],
                    "promedios": hum_avgs["valores"]
                },
                "velocidades": {
                    "maximas": vel_maxs["valores"],
                    "minimas": vel_mins["valores"],
                    "promedios": vel_avgs["valores"]
                }
            }
        except Exception as e:
            raise Exception(f"Error obteniendo datos climáticos para {city}: {str(e)}")


    def get_multiple_cities_climate(self, db: Session, search, cities_list) -> Dict[str, Any]:
        """
        Obtiene datos climáticos para múltiples ciudades de forma independiente
        """
        try:
            results = {}
            
            for city in cities_list:
                try:
                    city_data = self.get_climate_daily_all_types_per_city(db=db, search=search, city=city)
                    results[city] = city_data
                except Exception as e:
                    # Si una ciudad falla, continúa con las demás
                    results[city] = {
                        "error": f"No se pudieron obtener datos para {city}: {str(e)}"
                    }
            
            return {
                "ciudades_procesadas": len(cities_list),
                "ciudades_exitosas": len([r for r in results.values() if "error" not in r]),
                "resultados": results
            }
        except Exception as e:
            raise Exception(f"Error procesando múltiples ciudades: {str(e)}")
class CorrelationService():
    def get_correlatio_matrix(self, db: Session, search: SearchCorrelationSchema):
        # Correlacion con granuralidad de 24 horas
        climate = ClimateService()
        demands = DemandService()
        if search.tipo == 0:
            variables = climate.get_climate(db=db, search=SearchAllClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin))
            num_days = len(variables["lista_fechas"])
            num_rows = num_days * 24

            df = pd.DataFrame(columns=[], index=range(num_rows))

            if 1 in search.list_ids:
                temperatura = variables["lista_temperaturas"]
                df["temperatura"] = [item for sublist in temperatura for item in sublist]
                if temperatura == []:
                    raise Exception("No hay datos de temperatura para el rango de fechas seleccionado")
            if 2 in search.list_ids:
                humedad = variables["lista_humedades"]
                df["humedad"] = [item for sublist in humedad for item in sublist]
                if humedad == []:
                    raise Exception("No hay datos de humedad para el rango de fechas seleccionado")
            if 3 in search.list_ids:
                velocidad = variables["lista_velocidades"]
                df["velocidad"] = [item for sublist in velocidad for item in sublist]
                if velocidad == []:
                    raise Exception("No hay datos de velocidad para el rango de fechas seleccionado")
            if not any(option in search.list_ids for option in [1, 2, 3]):
                raise Exception("No se selecciono ningun tipo de variable valido")
            demand_data = demands.get_24_demands(db=db, search=DemandSearchSchema(tipo=0, fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin))
            demanda = demand_data["lista_demandas"]
            if demanda == []:
                raise Exception("No hay datos de demanda para el rango de fechas seleccionado")
            df["demanda"] = [item for sublist in demanda for item in sublist]
            
        # Correlacion con granuralidad de un dia
        elif search.tipo == 1:
            temperatura = climate.get_climate_day(db=db, search=SearchDayClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=0))
            lista_temperaturas = temperatura["lista_variables"]
            num_days = len(temperatura["lista_fechas"])
            df = pd.DataFrame(columns=[], index=range(num_days))
            if 1 in search.list_ids:
                df["temperatura"] = lista_temperaturas
                if lista_temperaturas == []:
                    raise Exception("No hay datos de temperatura para el rango de fechas seleccionado")
            if 2 in search.list_ids:
                humedad = climate.get_climate_day(db=db, search=SearchDayClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=1))
                lista_humedades = humedad["lista_variables"]
                df["humedad"] = lista_humedades
                if lista_humedades == []:
                    raise Exception("No hay datos de humedad para el rango de fechas seleccionado")
            if 3 in search.list_ids:
                velocidad = climate.get_climate_day(db=db, search=SearchDayClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=2))
                lista_velocidades = velocidad["lista_variables"]
                df["velocidad"] = lista_velocidades
                if lista_velocidades == []:
                    raise Exception("No hay datos de velocidad para el rango de fechas seleccionado")
            if not any(option in search.list_ids for option in [1, 2, 3]):
                raise Exception("No se selecciono ningun tipo de variable valido")
            demandas = demands.get_demands(db=db, search=DemandSearchSchema(tipo=1, fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin))
            demandas = demandas["lista_demandas"]   
            if demandas == []:
                raise Exception("No hay datos de demanda para el rango de fechas seleccionado")
            df["demanda"] = demandas

        # Correlacion con granuralidad de un mes
        elif search.tipo == 2:
            temperatura = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=0))
            lista_temperaturas = temperatura["lista_variables"]
            num_months = len(temperatura["lista_fechas"])
            df = pd.DataFrame(columns=[], index=range(num_months))
            if 1 in search.list_ids:
                df["temperatura"] = lista_temperaturas
                if lista_temperaturas == []:
                    raise Exception("No hay datos de temperatura para el rango de fechas seleccionado")
            if 2 in search.list_ids:
                humedad = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=1))
                lista_humedades = humedad["lista_variables"]
                df["humedad"] = lista_humedades
                if lista_humedades == []:
                    raise Exception("No hay datos de humedad para el rango de fechas seleccionado")
            if 3 in search.list_ids:
                velocidad = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=2))
                lista_velocidades = velocidad["lista_variables"]
                df["velocidad"] = lista_velocidades
                if lista_velocidades == []:
                    raise Exception("No hay datos de velocidad para el rango de fechas seleccionado")
            if not any(option in search.list_ids for option in [1, 2, 3]):
                raise Exception("No se selecciono ningun tipo de variable valido")
            demandas = demands.get_demands(db=db, search=DemandSearchSchema(tipo=2, fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin))
            demandas = demandas["lista_demandas"]   
            if demandas == []:
                raise Exception("No hay datos de demanda para el rango de fechas seleccionado")
            df["demanda"] = demandas

        # Correlacion con granuralidad de un ano
        elif search.tipo == 3:
            temperatura = climate.get_climate_year(db=db, search=SearchYearClimateSchema(ano_inicio=search.ano_inicio, ano_fin=search.ano_fin, tipo=0, id=0))
            lista_temperaturas = temperatura["lista_variables"]
            num_years = len(temperatura["lista_fechas"])
            df = pd.DataFrame(columns=[], index=range(num_years))
            if search.list_id_macroeconomic is not None:
                macroeconomic = MacroeconomicService()
                macroeconomic_ids = search.list_id_macroeconomic
                for id in macroeconomic_ids:
                    name = db.query(Macroeconomics.name).filter(Macroeconomics.id == id).all()
                    values = macroeconomic.get_variable_year(db=db, id=id, fecha_inicio=search.ano_inicio, fecha_fin=search.ano_fin)
                    if values["lista_variables"] == []:
                        raise Exception("No hay datos de la variable seleccionada para el rango de fechas seleccionado")
                    if len(values["lista_variables"]) != num_years:
                        years_empty = [year for year in range(int(search.ano_inicio), int(search.ano_fin)+1) if year not in values["lista_fechas"]]
                        raise Exception(f"No hay datos de la variable {name[0][0]} para todas las fechas seleccionadas. Anos sin datos: {years_empty}")
                    df[name[0][0]] = values["lista_variables"]
            if search.list_ids is not None:
                if 1 in search.list_ids:
                    df["temperatura"] = lista_temperaturas
                    if lista_temperaturas == []:
                        raise Exception("No hay datos de temperatura para el rango de fechas seleccionado")
                if 2 in search.list_ids:
                    humedad = climate.get_climate_year(db=db, search=SearchYearClimateSchema( ano_inicio=search.ano_inicio, ano_fin=search.ano_fin, tipo=0, id=1))
                    lista_humedades = humedad["lista_variables"]
                    df["humedad"] = lista_humedades
                    if lista_humedades == []:
                        raise Exception("No hay datos de humedad para el rango de fechas seleccionado")
                if 3 in search.list_ids:
                    velocidad = climate.get_climate_year(db=db, search=SearchYearClimateSchema(ano_inicio=search.ano_inicio, ano_fin=search.ano_fin, tipo=0, id=2))
                    lista_velocidades = velocidad["lista_variables"]
                    df["velocidad"] = lista_velocidades
                    if lista_velocidades == []:
                        raise Exception("No hay datos de velocidad para el rango de fechas seleccionado")
                if not any(option in search.list_ids for option in [1, 2, 3]):
                    raise Exception("No se selecciono ningun tipo de variable valido")
            demandas = demands.get_demands(db=db, search=DemandSearchSchema(tipo=3, ano_inicio=search.ano_inicio, ano_fin=search.ano_fin))
            demandas = demandas["lista_demandas"]  
            print("demandas", demandas) 
            if demandas == []:
                raise Exception("No hay datos de demanda para el rango de fechas seleccionado")
            df["demanda"] = demandas
            print("df", df)


        correlation_matrix = df.corr().values.tolist()
        return {
            "correlation_matrix": correlation_matrix,
            "correlation_matrix_labels": df.columns.tolist()
        }


class GridAnalysisService():
    def get_number_of_days(self, year, month):
        number_of_days = calendar.monthrange(year, month)[1]
        number_of_saturdays = sum(1 for day in range(1, number_of_days + 1) if calendar.weekday(year, month, day) == calendar.SATURDAY)
        number_of_sundays = sum(1 for day in range(1, number_of_days + 1) if calendar.weekday(year, month, day) == calendar.SUNDAY)
        holidays = get_colombia_holidays_by_year(year)
        holidays = list(holidays)

        holidays = [holiday.date.strftime('%Y-%m-%d') for holiday in holidays]

        number_of_holidays_on_month = 0
        for holiday in holidays:
            if holiday.startswith(f"{year}-0{month}"):
                number_of_holidays_on_month += 1
        number_of_working_days = number_of_days - number_of_saturdays - number_of_sundays - number_of_holidays_on_month
        month_total_days = number_of_working_days + number_of_saturdays + number_of_sundays + number_of_holidays_on_month
        return number_of_working_days, number_of_saturdays, number_of_sundays, number_of_holidays_on_month, month_total_days

    def get_days(self, db: Session, search: SearchAnalysisGridSchema):
        try:
            start_date = datetime.strptime(search.fecha_inicio, '%Y-%m-%d')
            end_date = datetime.strptime(search.fecha_fin, '%Y-%m-%d')
            date_list = []
            current_date = start_date
            while current_date <= end_date:
                date_list.append(current_date.strftime('%Y-%m'))
                current_date = current_date.replace(day=28) + timedelta(days=4)
                current_date = current_date.replace(day=1)
            
            total_month_days = []
            total_working_days = []
            total_saturdays = []
            total_sundays = []
            total_holidays = []

            set_dates = set(date_list)
            for date in sorted(set_dates):
                year = int(date[:4])
                month = int(date[5:])
                number_of_working_days, number_of_saturdays, number_of_sundays, number_of_holidays_on_month, month_total_days = self.get_number_of_days(year, month)
                total_month_days.append(month_total_days)
                total_working_days.append(number_of_working_days)
                total_saturdays.append(number_of_saturdays)
                total_sundays.append(number_of_sundays)
                total_holidays.append(number_of_holidays_on_month)

            return {
                "dates": sorted(set_dates),
                "total_month_days": total_month_days,
                "total_working_days": total_working_days,
                "total_saturdays": total_saturdays,
                "total_sundays": total_sundays,
                "total_holidays": total_holidays
            }
        except Exception as e:
            raise e

        

    
    def get_grid_info(self, db: Session, search: SearchAnalysisGridSchema):
        # Monthly data
        try:
            demands = DemandService()
            monthly_demand = demands.get_demands(db=db, search=DemandSearchSchema(tipo=2, fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin))
            dates = monthly_demand["lista_fechas"]
            monthly_demand = monthly_demand["lista_demandas"]
            climate = ClimateService()
            min_temp = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=2, id=0))
            min_temp = min_temp["lista_variables"]
            max_temp = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=1, id=0))
            max_temp = max_temp["lista_variables"]
            avg_temp = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=0))
            avg_temp = avg_temp["lista_variables"]

            min_hum = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=2, id=1))
            min_hum = min_hum["lista_variables"]
            max_hum = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=1, id=1))
            max_hum = max_hum["lista_variables"]
            avg_hum = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=1))
            avg_hum = avg_hum["lista_variables"]

            min_vel = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=2, id=2))
            min_vel = min_vel["lista_variables"]
            max_vel = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=1, id=2))
            max_vel = max_vel["lista_variables"]
            avg_vel = climate.get_climate_month(db=db, search=SearchMonthClimateSchema(fecha_inicio=search.fecha_inicio, fecha_fin=search.fecha_fin, tipo=0, id=2))
            avg_vel = avg_vel["lista_variables"]

            total_month_days = []
            total_working_days = []
            total_saturdays = []
            total_sundays = []
            total_holidays = []
            for i in range(len(dates)):
                year = int(dates[i][:4])
                month = int(dates[i][5:])
                number_of_working_days, number_of_saturdays, number_of_sundays, number_of_holidays_on_month, month_total_days = self.get_number_of_days(year, month)
                total_month_days.append(month_total_days)
                total_working_days.append(number_of_working_days)
                total_saturdays.append(number_of_saturdays)
                total_sundays.append(number_of_sundays)
                total_holidays.append(number_of_holidays_on_month)
            
            return {
                "dates": dates,
                "monthly_demand": monthly_demand,
                "min_temp": min_temp,
                "max_temp": max_temp,
                "avg_temp": avg_temp,
                "min_hum": min_hum,
                "max_hum": max_hum,
                "avg_hum": avg_hum,
                "min_vel": min_vel,
                "max_vel": max_vel,
                "avg_vel": avg_vel,
                "total_month_days": total_month_days,
                "total_working_days": total_working_days,
                "total_saturdays": total_saturdays,
                "total_sundays": total_sundays,
                "total_holidays": total_holidays
            }


        except SQLAlchemyError as e:
            raise Exception(f"Error al obtener datos de demanda: {e}")
        except Exception as e:
            raise e
        

