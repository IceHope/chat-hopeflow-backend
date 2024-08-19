import datetime
from typing import Optional

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dao.users_dao import UserDAO

user_router = APIRouter()

# JWT 配置
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

user_dao = UserDAO()


# Pydantic models for request bodies
class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserForgotPassword(BaseModel):
    username: str


class Token(BaseModel):
    access_token: str
    token_type: str


# Function to create JWT
def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Register endpoint
@user_router.post("/user/register")
def register(user: UserCreate):
    existing_user = user_dao.get_user_by_name(user.username)

    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    try:
        user_dao.add_new_user(name=user.username, password=user.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "User registered successfully"}


# Login endpoint
@user_router.post("/user/login")
def login(user: UserLogin):
    db_user = user_dao.get_user_by_name(user.username)

    if not db_user:
        raise HTTPException(status_code=400, detail="Username not found. Please register.")

    if db_user.password != user.password:
        raise HTTPException(status_code=400, detail="Incorrect password")

    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Forgot password endpoint
@user_router.post("/user/forgot_password")
def forgot_password(user: UserForgotPassword):
    db_user = user_dao.get_user_by_name(user.username)

    if not db_user:
        raise HTTPException(status_code=400, detail="Username not found. Please register.")

    return {"password": db_user.password}
