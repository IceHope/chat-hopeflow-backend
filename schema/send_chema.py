import json
import pprint
from enum import Enum

from pydantic import BaseModel

RAG_RETRIEVE_CHUNK = "retrieve chunk "
RAG_EVENT_IMAGE_QA = "generate image response"
RAG_EVENT_GENERATE = "generate final response"


class SendType(str, Enum):
    SUGGEST_QUESTION = "suggest_question"
    RAG_EVENT = "rag_event"
    CHAT_STREAM_STATUS = "chat_stream_status"
    CHAT_STREAM_CONTENT = "chat_stream_content"


class SendSchema(BaseModel):
    send_type: str
    send_data: str


if __name__ == "__main__":
    send_schema = SendSchema(
        send_type=SendType.CHAT_STREAM_CONTENT.value, send_data="stop"
    )
    send = json.dumps(send_schema.model_dump_json())
    pprint.pprint(send)
