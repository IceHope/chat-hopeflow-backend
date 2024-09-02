from pydantic import BaseModel


class UserNameRequest(BaseModel):
    user_name: str


class FileIdKnowledgeRequest(BaseModel):
    file_id: str


class FileIdChunkRequest(BaseModel):
    file_id: str
    query: str


class ChatQueryRequest(BaseModel):
    query: str
    file_id: str
