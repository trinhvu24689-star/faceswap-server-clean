import os
from datetime import datetime, date

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Date,
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./swap_history.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SwapHistoryModel(Base):
    __tablename__ = "swap_history"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), index=True)
    user_id = Column(String(128), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    source_path = Column(String(512))
    target_path = Column(String(512))
    result_path = Column(String(512))

    status = Column(String(32), default="success")
    message = Column(String(512), default="")

    credits_used = Column(Integer, default=1)
    client_ip = Column(String(64), default="")
    user_agent = Column(String(256), default="")

    billing_day = Column(Date, default=date.today)


def init_swap_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
