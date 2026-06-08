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
    Float,
    UniqueConstraint
)
from datetime import datetime
from api.config.database import Base, engine
from sqlalchemy.orm import relationship


class SessionGMR(Base):
    __tablename__ = "GMR_session"
    id = Column(Integer, Sequence('gmr_session_id_seq'), primary_key=True)
    name = Column(String(300))
    user_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.now())
    updated_at = Column(DateTime, default=datetime.now())
    is_active = Column(Boolean, default=True)
    description = Column(String(300), nullable=True)
    user_models = relationship("UsersModels", cascade="all, delete-orphan", backref="session")
    type_year = relationship("TypeYear", cascade="all, delete-orphan", backref="session")
class UsersModels(Base):
    __tablename__ = "SphaerAI_users_models"
    id = Column(Integer, Sequence('gmr_users_models_id_seq'), primary_key=True)
    model_name = Column(String(100))
    user_id = Column(Integer)
    session_id = Column(Integer, ForeignKey('GMR_session.id', ondelete="CASCADE"))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now())
    updated_at = Column(DateTime, default=datetime.now())
    __table_args__ = (UniqueConstraint('model_name', 'user_id', 'session_id', name='_user_model_name_session_uc'),)
    values = relationship("UsersModelsValues", cascade="all, delete-orphan", backref="model")

class UsersModelsValues(Base):
    __tablename__ = "SphaerAI_users_models_values"
    id = Column(Integer, Sequence('gmr_users_models_values_id_seq'), primary_key=True)
    model_id = Column(Integer, ForeignKey('SphaerAI_users_models.id', ondelete="CASCADE"))
    date = Column(DateTime)
    value = Column(Float)
    climate_type = Column(Enum("NORMAL", "NIÑO", "NIÑA"), default="NORMAL")



