from models.factory.llm_factory import LLMFactory
from models.model_type import LLMType
from rag.config.rag_config import RagConfiguration
from utils.log_utils import LogUtils

RAG_MULTI_MODAL = [
    ("gpt-4o", LLMType.OPENAI),
    ("gpt-4o-mini", LLMType.OPENAI),
    ("qwen-vl-max", LLMType.DASHSCOPE, 20, 20),
    ("qwen-vl-plus", LLMType.DASHSCOPE, 8, 8),
    ("gemini-1.5-pro", LLMType.GOOGLE, 25.55, 76.65),
    ("gemini-1.5-flash", LLMType.GOOGLE, 0.55, 2.19),
]


def _get_mutil_modal_item(mode_name):
    for item in RAG_MULTI_MODAL:
        if item[0] == mode_name:
            return item
    raise ValueError(f"Unknown RAG multi-modal: {mode_name}")


def get_mutil_modal_model(model_name: str):
    LogUtils.log_info(f"Rag multi-modal: {model_name}")

    modal_item = _get_mutil_modal_item(model_name)

    return LLMFactory.get_llm(modal_item[1], modal_item[0])


def get_mutil_modal_config_model():
    _config_modal_name = RagConfiguration().get_multi_modal_config()
    return get_mutil_modal_model(_config_modal_name)