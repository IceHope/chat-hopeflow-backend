import time
import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, BigInteger, String, Text

from dao.db_config import Base, get_db, engine
from utils.log_utils import LogUtils


####################
# User DB Schema
####################

class User(Base):
    __tablename__ = "user"

    id = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)  # Ensure name is unique
    password = Column(Text, nullable=False)
    last_active_at = Column(BigInteger)
    updated_at = Column(BigInteger)
    created_at = Column(BigInteger)


# Pydantic model for user
class UserModel(BaseModel):
    id: str
    name: str
    password: str
    last_active_at: int  # timestamp in epoch
    updated_at: int  # timestamp in epoch
    created_at: int  # timestamp in epoch

    model_config = ConfigDict(from_attributes=True)


# DAO class


class UserDAO:
    def __init__(self):
        # Initialize the database tables
        self._initialize_database()

    def _initialize_database(self):
        # Create tables based on the models if they don't exist
        Base.metadata.create_all(bind=engine)

    def add_new_user(
            self,
            name: str,
            password: str
    ) -> Optional[UserModel]:
        with get_db() as db:
            user = UserModel(
                id=str(uuid.uuid4()),
                name=name,
                password=password,
                last_active_at=int(time.time()),
                created_at=int(time.time()),
                updated_at=int(time.time()),
            )

            result = User(**user.model_dump())
            db.add(result)
            db.commit()
            db.refresh(result)
            return user

    def get_user_by_name(self, name: str) -> Optional[UserModel]:
        """Get user by username."""
        with get_db() as db:
            user = db.query(User).filter_by(name=name).first()
            if user:
                return UserModel.model_validate(user)
            return None

    def validate_login(self, name: str, password: str) -> Optional[UserModel]:
        """Validate login with username and password."""
        with get_db() as db:
            user = db.query(User).filter_by(name=name).first()
            if not user:
                LogUtils.log_error(f"Username does not exist: {name}")
                raise ValueError("Username does not exist. Please register.")

            if user.password != password:
                LogUtils.log_error(f"Incorrect password for user: {name}")
                raise ValueError("Incorrect password.")

            # Update last active time
            user.last_active_at = int(time.time())

            db.commit()  # Commit the changes to the database

            return UserModel.model_validate(user)
