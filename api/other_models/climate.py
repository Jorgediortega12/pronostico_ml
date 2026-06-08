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


class Climate(Base):
    __tablename__ = "GMR_climates"
    id = Column(Integer, Sequence('gmr_climates_id_seq'), primary_key=True)
    t1 = Column(Float)
    t2 = Column(Float)
    t3 = Column(Float)
    t4 = Column(Float)
    t5 = Column(Float)
    t6 = Column(Float)
    t7 = Column(Float)
    t8 = Column(Float)
    t9 = Column(Float)
    t10 = Column(Float)
    t11 = Column(Float)
    t12 = Column(Float)
    t13 = Column(Float)
    t14 = Column(Float)
    t15 = Column(Float)
    t16 = Column(Float)
    t17 = Column(Float)
    t18 = Column(Float)
    t19 = Column(Float)
    t20 = Column(Float)
    t21 = Column(Float)
    t22 = Column(Float)
    t23 = Column(Float)
    t24 = Column(Float)
    h1 = Column(Float)
    h2 = Column(Float)
    h3 = Column(Float)
    h4 = Column(Float)
    h5 = Column(Float)
    h6 = Column(Float)
    h7 = Column(Float)
    h8 = Column(Float)
    h9 = Column(Float)
    h10 = Column(Float)
    h11 = Column(Float)
    h12 = Column(Float)
    h13 = Column(Float)
    h14 = Column(Float)
    h15 = Column(Float)
    h16 = Column(Float)
    h17 = Column(Float)
    h18 = Column(Float)
    h19 = Column(Float)
    h20 = Column(Float)
    h21 = Column(Float)
    h22 = Column(Float)
    h23 = Column(Float)
    h24 = Column(Float)
    v1 = Column(Float)
    v2 = Column(Float)
    v3 = Column(Float)
    v4 = Column(Float)
    v5 = Column(Float)
    v6 = Column(Float)
    v7 = Column(Float)
    v8 = Column(Float)
    v9 = Column(Float)
    v10 = Column(Float)
    v11 = Column(Float)
    v12 = Column(Float)
    v13 = Column(Float)
    v14 = Column(Float)
    v15 = Column(Float)
    v16 = Column(Float)
    v17 = Column(Float)
    v18 = Column(Float)
    v19 = Column(Float)
    v20 = Column(Float)
    v21 = Column(Float)
    v22 = Column(Float)
    v23 = Column(Float)
    v24 = Column(Float)
    fecha = Column(DateTime, default=datetime.utcnow, nullable=False)






















