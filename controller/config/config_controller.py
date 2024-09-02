from fastapi import APIRouter
from typing import Dict

config_router = APIRouter()

from utils.constants import (
    COMMAND_DONE_FROM_SERVE,
    COMMAND_STOP_FROM_CLIENT,
    COMMAND_STREAM_START_FROM_SERVE,
    RAG_RETRIEVE_CHUNK_START,
    RAG_RETRIEVE_CHUNK_DONE,
    RAG_RERANK_CHUNK_START,
    RAG_RERANK_CHUNK_DONE,
    RAG_EVENT_IMAGE_QA_START,
    RAG_EVENT_IMAGE_QA_DONE,
    RAG_EVENT_GENERATE_START,
)


@config_router.get("/config/command")
async def get_command_config() -> Dict[str, str]:
    command_config = {
        "command_done_from_serve": COMMAND_DONE_FROM_SERVE,
        "command_stop_from_client": COMMAND_STOP_FROM_CLIENT,
        "command_stream_start_from_serve": COMMAND_STREAM_START_FROM_SERVE,
        "rag_retrieve_chunk_start": RAG_RETRIEVE_CHUNK_START,
        "rag_retrieve_chunk_done": RAG_RETRIEVE_CHUNK_DONE,
        "rag_rerank_chunk_start": RAG_RERANK_CHUNK_START,
        "rag_rerank_chunk_done": RAG_RERANK_CHUNK_DONE,
        "rag_event_image_qa_start": RAG_EVENT_IMAGE_QA_START,
        "rag_event_image_qa_done": RAG_EVENT_IMAGE_QA_DONE,
        "rag_event_generate_start": RAG_EVENT_GENERATE_START,
    }
    return command_config
