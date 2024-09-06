from fastapi import APIRouter
from typing import Dict

from schema.command_shema import CommandSchema, get_command_schema
from schema.global_config_shema import GlobalConfigSchema
from schema.model_schema import (
    get_all_embedding_models,
    get_all_multimodal_models,
    get_all_llm_models,
    ModelListSchema, get_model_list_schema,
)
from utils.command_constants import *

config_router = APIRouter()


@config_router.get("/config/command")
async def get_command_config() -> CommandSchema:
    return get_command_schema()


@config_router.get("/config/models/llm")
async def get_llm_models():
    return get_all_llm_models()


@config_router.get("/config/models/multimodal")
async def get_multimodal_models():
    return get_all_multimodal_models()


@config_router.get("/config/models/embedding")
async def get_embedding_models():
    return get_all_embedding_models()


@config_router.get("/config/models/rerank")
async def get_all_rerank_models():
    return get_all_embedding_models()


@config_router.get("/config/global")
async def get_global_config():
    return GlobalConfigSchema(
        commands=get_command_schema(),
        models=get_model_list_schema()
    )
