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


class Macroeconomics(Base):
    __tablename__ = "PRONOSTICO_macroeconomics"
    id = Column(Integer, Sequence('pronostico_macroeconomics_id_seq'), primary_key=True)
    name = Column(String(250))

class MacroeconomicsData(Base):
    __tablename__ = "PRONOSTICO_macroeconomics_data"
    id = Column(Integer, Sequence('gmr_macroeconomics_data_id_seq'), primary_key=True)
    ano = Column(Integer)          # ← sin tilde
    eco_id = Column(Integer, ForeignKey('PRONOSTICO_macroeconomics.id'))  # ← también corrige el FK
    value = Column(Float)
