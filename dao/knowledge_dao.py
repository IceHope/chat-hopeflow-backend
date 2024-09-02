import time
import uuid
from typing import Optional, List

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, BigInteger, Integer, String, Text

from dao.db_config import Base, get_db, engine


class Knowledge(Base):
    """知识库模型定义,一个文件一个对象"""

    __tablename__ = "knowledge"

    id = Column(String, primary_key=True)
    user_name = Column(String, nullable=False, comment="用户名")
    file_id = Column(String, unique=True, nullable=False, comment="文件id")
    file_path = Column(Text, unique=True, nullable=False, comment="文件路径")
    file_name = Column(Text, nullable=False, comment="文件名")
    file_size = Column(BigInteger, comment="文件大小")
    chunk_size = Column(Integer, comment="块大小")
    chunk_overlap = Column(Integer, comment="块重叠")
    file_title = Column(Text, nullable=False, comment="文件标题")
    updated_at = Column(BigInteger, comment="更新时间")
    created_at = Column(BigInteger, comment="创建时间")


class KnowledgeModel(BaseModel):
    """知识模型定义"""

    id: str
    file_id: str
    user_name: str
    file_path: str
    file_name: str
    file_size: int
    chunk_size: int
    chunk_overlap: int
    file_title: str
    updated_at: int
    created_at: int

    model_config = ConfigDict(from_attributes=True)


class KnowledgeDao:
    """知识数据访问对象"""

    def __init__(self):
        self._initialize_database()

    def _initialize_database(self):
        """初始化数据库表"""
        Base.metadata.create_all(bind=engine)

    def add_new_knowledge(
        self,
        user_name: str,
        file_id: str,
        file_path: str,
        file_name: str,
        file_size: int,
        file_title: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> Optional[KnowledgeModel]:
        """添加新知识"""
        _file_title = file_name if file_title is None else file_title
        with get_db() as db:
            existing_knowledge = (
                db.query(Knowledge).filter_by(file_path=file_path).first()
            )
            if existing_knowledge:
                # 如果文件路径已存在，更新知识
                existing_knowledge.user_name = user_name
                existing_knowledge.file_id = file_id
                existing_knowledge.file_name = file_name
                existing_knowledge.file_size = file_size
                existing_knowledge.chunk_size = chunk_size
                existing_knowledge.chunk_overlap = chunk_overlap
                existing_knowledge.file_title = _file_title
                existing_knowledge.updated_at = int(time.time())
                db.commit()
                db.refresh(existing_knowledge)
                return KnowledgeModel.model_validate(existing_knowledge)
            else:
                # 如果文件路径不存在，添加新知识
                knowledge_model = KnowledgeModel(
                    id=str(uuid.uuid4()),
                    user_name=user_name,
                    file_id=file_id,
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    file_title=_file_title,
                    created_at=int(time.time()),
                    updated_at=int(time.time()),
                )
                result = Knowledge(**knowledge_model.model_dump())
                db.add(result)
                db.commit()
                db.refresh(result)
                return knowledge_model

    def get_all_knowledges_by_user_name(self, user_name: str) -> List[KnowledgeModel]:
        """通过用户名获取知识列表"""
        with get_db() as db:
            knowledge_list = db.query(Knowledge).filter_by(user_name=user_name).all()
            if knowledge_list:
                return [
                    KnowledgeModel.model_validate(knowledge)
                    for knowledge in knowledge_list
                ]
            return []

    def get_all_knowledges_by_admin(self) -> List[KnowledgeModel]:
        """通过管理员获取知识列表"""
        return self.get_all_knowledges_by_user_name("admin")

    def update_knowledge_by_file_id(
        self, file_id: str, file_title: str
    ) -> Optional[KnowledgeModel]:
        """通过id更新知识"""
        with get_db() as db:
            knowledge = db.query(Knowledge).filter_by(file_id=file_id).first()
            if knowledge:
                knowledge.file_title = file_title
                knowledge.updated_at = int(time.time())
                db.commit()
                db.refresh(knowledge)
                return KnowledgeModel.model_validate(knowledge)
            return None

    def delete_knowledge_by_file_id(self, file_id: str) -> Optional[KnowledgeModel]:
        """通过file_id删除知识"""
        with get_db() as db:
            knowledge = db.query(Knowledge).filter_by(file_id=file_id).first()
            if knowledge:
                db.delete(knowledge)
                db.commit()
                return KnowledgeModel.model_validate(knowledge)
            return None

    def delete_all_knowledges_by_user_name(
        self, user_name: str
    ) -> List[KnowledgeModel]:
        """通过用户名删除所有知识库"""
        with get_db() as db:
            knowledges = db.query(Knowledge).filter_by(user_name=user_name).all()
            if knowledges:
                deleted_knowledges = []
                for knowledge in knowledges:
                    db.delete(knowledge)
                    deleted_knowledges.append(KnowledgeModel.model_validate(knowledge))
                db.commit()
                return deleted_knowledges
            return []
