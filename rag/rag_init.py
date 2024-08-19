import os
import time

from llama_index.core import Settings
from dotenv import load_dotenv

from utils.log_utils import LogUtils

load_dotenv()


class RagInit:
    @staticmethod
    def init():
        RagInit._init_llm()
        RagInit._init_embedding()

    @staticmethod
    def _init_llm():
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.7,
        )

    @staticmethod
    def _init_embedding():
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        start_time = time.time()
        Settings.embed_model = HuggingFaceEmbedding(model_name="F:/HuggingFace/Embedding/bge-m3")
        LogUtils.log_info(f"Embedding model loaded in {time.time() - start_time} seconds")
