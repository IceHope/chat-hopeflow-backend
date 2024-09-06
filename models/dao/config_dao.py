import time
import uuid
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, BigInteger, Integer, String, Text

from dao.db_config import Base, get_db, engine


class Config(Base):
    __tablename__ = "config"

    id = Column(String, primary_key=True)
    user_name = Column(String, nullable=False, comment="用户名")
    rag_similarity_top_k = Column(Integer, default=3)
    # 未来可以在这里添加更多配置属性


class ConfigModel(BaseModel):
    id: str
    user_name: str
    rag_similarity_top_k: int = 3
    # 未来可以在这里添加更多配置属性

    model_config = ConfigDict(from_attributes=True)


class ConfigDao:
    def __init__(self):
        self._initialize_database()

    def _initialize_database(self):
        """初始化数据库表"""
        Base.metadata.create_all(bind=engine)

    def initialize_user_config(self, user_name: str) -> ConfigModel:
        """在用户登录时初始化配置"""
        with get_db() as db:
            existing_config = db.query(Config).filter_by(user_name=user_name).first()
            if existing_config:
                return ConfigModel.model_validate(existing_config)

            new_config = Config(
                id=str(uuid.uuid4()),
                user_name=user_name,
                rag_similarity_top_k=3,  # 默认值
                # 未来可以在这里添加更多配置属性的默认值
            )
            db.add(new_config)
            db.commit()
            db.refresh(new_config)
            return ConfigModel.model_validate(new_config)

    def update_config(self, user_name: str, **kwargs: Any) -> ConfigModel:
        """更新用户配置"""
        with get_db() as db:
            config = db.query(Config).filter_by(user_name=user_name).first()
            if not config:
                raise ValueError(f"未找到用户 {user_name} 的配置")

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    raise ValueError(f"配置项 {key} 不存在")

            db.commit()
            db.refresh(config)
            return ConfigModel.model_validate(config)
