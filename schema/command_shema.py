from pydantic import BaseModel

from utils.command_constants import *


class CommandSchema(BaseModel):
    chat_stream_serve_start: str
    chat_stream_serve_done: str
    chat_stream_client_stop: str
    rag_parse_question_start: str
    rag_parse_question_done: str
    rag_retrieve_chunk_start: str
    rag_retrieve_chunk_done: str
    rag_rerank_chunk_start: str
    rag_rerank_chunk_done: str
    rag_event_image_qa_start: str
    rag_event_image_qa_done: str
    rag_event_generate_start: str


def get_command_schema():
    return CommandSchema(
        chat_stream_serve_start=CHAT_STREAM_SERVE_START,
        chat_stream_serve_done=CHAT_STREAM_SERVE_DONE,
        chat_stream_client_stop=CHAT_STREAM_CLIENT_STOP,
        rag_parse_question_start=RAG_PARSE_QUESTION_START,
        rag_parse_question_done=RAG_PARSE_QUESTION_DONE,
        rag_retrieve_chunk_start=RAG_RETRIEVE_CHUNK_START,
        rag_retrieve_chunk_done=RAG_RETRIEVE_CHUNK_DONE,
        rag_rerank_chunk_start=RAG_RERANK_CHUNK_START,
        rag_rerank_chunk_done=RAG_RERANK_CHUNK_DONE,
        rag_event_image_qa_start=RAG_EVENT_IMAGE_QA_START,
        rag_event_image_qa_done=RAG_EVENT_IMAGE_QA_DONE,
        rag_event_generate_start=RAG_EVENT_GENERATE_START,
    )
