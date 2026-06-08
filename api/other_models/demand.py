from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    DateTime,
    JSON,
    Table, 
    func,
    Sequence,
    Text,
    Enum,
    Float
)
from datetime import datetime
from api.config.database import Base, engine
from sqlalchemy.orm import relationship



class Demand(Base):
    __tablename__ = "PRONOSTICO_demands"
    id = Column(Integer, Sequence('gmr_demands_id_seq'), primary_key=True)
    hora_1 = Column(Float)
    hora_2 = Column(Float)
    hora_3 = Column(Float)
    hora_4 = Column(Float)
    hora_5 = Column(Float)
    hora_6 = Column(Float)
    hora_7 = Column(Float)
    hora_8 = Column(Float)
    hora_9 = Column(Float)
    hora_10 = Column(Float)
    hora_11 = Column(Float)
    hora_12 = Column(Float)
    hora_13 = Column(Float)
    hora_14 = Column(Float)
    hora_15 = Column(Float)
    hora_16 = Column(Float)
    hora_17 = Column(Float)
    hora_18 = Column(Float)
    hora_19 = Column(Float)
    hora_20 = Column(Float)
    hora_21 = Column(Float)
    hora_22 = Column(Float)
    hora_23 = Column(Float)
    hora_24 = Column(Float)
    total = Column(Float)
    fecha = Column(DateTime, default=datetime.utcnow, nullable=False)
    tipo_fecha = Column(Enum("0", "1", "2"), default="0") 
    file_type = Column(String(3), default="txf")


class YearlyDemand(Base):
    __tablename__ = "SphaerAI_yearly_demand"
    id = Column(Integer, Sequence('gmr_yearly_demand_id_seq'), primary_key=True)
    demand = Column(Float)
    year = Column(Integer)

class MonthlyDemand(Base):
    __tablename__ = "SphaerAI_monthly_demand"
    id = Column(Integer, Sequence('gmr_monthly_demand_id_seq'), primary_key=True)
    value = Column(Float)
    year = Column(Integer)
    month = Column(Integer)
    percentage = Column(Float)
    climate_type = Column(Enum("NORMAL", "NIÑO", "NIÑA"), default="NORMAL")

class TypeYear(Base):
    __tablename__ = "SphaerAI_type_year"
    id = Column(Integer, Sequence('gmr_type_year_id_seq'), primary_key=True)
    year = Column(Integer)
    type = Column(Enum("TIPICO", "ATIPICO"), default="TIPICO")
    user_id = Column(Integer)   
    session_id = Column(Integer, ForeignKey("GMR_session.id", ondelete="CASCADE"))

class LastDemandDocument(Base):
    __tablename__ = "GMR_last_demand_document"
    id = Column(Integer, Sequence('gmr_last_demand_document_id_seq'), primary_key=True)
    document_type = Column(Enum("txf", "txr", "tx2"), default="txf")
    document_route = Column(Text)	
    document_date = Column(DateTime)











