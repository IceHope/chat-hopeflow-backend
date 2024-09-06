import os
import time
from typing import List, Tuple

from dotenv import load_dotenv
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilters

from dao.knowledge_dao import KnowledgeDao
from models.llm.llamaindex.groq_llm import GroqLlmaFactory
from rag.managers.chunk_manager import ChunkManager
from rag.managers.embedding_manager import EmbeddingManager
from rag.managers.generate_manager import GenerateManager
from rag.managers.image_qa_manger import ImageNodeQAManager
from rag.managers.query_manager import QueryManager
from rag.managers.reader_manager import ReaderManager
from rag.managers.rerank_manager import RerankManager
from rag.managers.retriever_manager import RetrieverManager
from rag.managers.vector_store_manager import VectorStoreManager
from rag.rag_utils import get_db_collection_name
from schema.rag_config import RagFrontendConfig
from utils.log_utils import LogUtils

load_dotenv()
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])


def check_image_node(nodes: List[NodeWithScore]):
    image_nodes: list[NodeWithScore] = []
    text_nodes: list[NodeWithScore] = []
    for node in nodes:
        metadata = node.metadata
        if "image_type" in metadata and metadata["image_type"] == "pdf_image":
            image_nodes.append(node)
        else:
            text_nodes.append(node)

    LogUtils.log_info(f"{len(text_nodes)} text_nodes, {len(image_nodes)} image_nodes")
    return text_nodes, image_nodes


class RagBaseManager:

    def __init__(self, user_name: str = "admin_test") -> None:
        """初始化所有Verba组件。"""
        LogUtils.log_info("开始初始化HopeManager")

        start_time = time.time()

        self.user_name = user_name

        Settings.llm = GroqLlmaFactory().get_llm()

        self.query_manager = QueryManager()
        # 初始化数据访问对象
        self.knowledge_dao = KnowledgeDao()

        # 初始化各个管理器
        self.reader_manager = ReaderManager()
        self.chunk_manager = ChunkManager()
        self.embedding_manager = EmbeddingManager()

        Settings.embed_model = self.embedding_manager.get_model()

        # 获取嵌入向量大小
        self.embedding_size = self.embedding_manager.get_dim()

        # 设置数据库集合名称
        self.db_collection_name = get_db_collection_name(
            self.embedding_manager.get_simple_model_name()
        )

        # 初始化向量存储管理器
        self.vector_store_manager = VectorStoreManager(
            collection_name=self.db_collection_name,
            embedding_size=self.embedding_size,
        )

        self.retriever_manager = RetrieverManager(
            vector_store=self.vector_store_manager.get_vector_store(),
        )
        self.rerank_manager = RerankManager()
        self.image_qa_manager = ImageNodeQAManager()
        self.generate_manager = GenerateManager()

        elapsed_time = round(time.time() - start_time, 2)  # 计算耗时
        LogUtils.log_info(f"HopeManager初始化完成,耗时: {elapsed_time}秒")

    def parse_context_question(self, origin_query: str, context: str):
        return self.query_manager.parse_context_question(
            origin_query=origin_query, context=context
        )

    def get_collection_name(self):
        return self.db_collection_name

    def auto_load_file_dir(self, file_dir: str) -> None:
        self._process_documents(self.reader_manager.load_file_dir(file_dir, True))

    def auto_load_file_list(self, file_list: list[str]) -> None:
        self._process_documents(self.reader_manager.load_file_list(file_list, True))

    # 先调用ReaderManager的manual_load_pdf方法,本地提取所有图片,然后人工清洗过滤,然后下面的方法,就不需要再次提取pdf的图片了
    def manual_load_file_dir(self, file_dir: str) -> None:
        self._process_documents(self.reader_manager.load_file_dir(file_dir, False))

    def manual_load_file_list(self, file_list: list[str]) -> None:
        self._process_documents(self.reader_manager.load_file_list(file_list, False))

    def _process_documents(self, documents: list[Document]) -> None:
        """
        处理文档：分块、向量化并存储。

        Args:
            documents: 文档列表
        """
        LogUtils.log_info(f"开始chunk {len(documents)} 个文档")

        # 文档分块
        nodes = self.chunk_manager.chunk_documents(documents)

        # 加载节点到向量存储
        self.vector_store_manager.load_nodes(nodes)

        # 将文档信息添加到知识库
        # 创建一个临时集合来存储已处理的file_id
        processed_file_ids = set()

        for document_item in documents:
            metadata = document_item.metadata
            if "image_type" in metadata and metadata["image_type"] == "pdf_image":
                continue

            file_id = metadata["file_id"]
            if file_id not in processed_file_ids:
                self.knowledge_dao.add_new_knowledge(
                    user_name=self.user_name,
                    file_id=file_id,
                    file_path=metadata["file_path"],
                    file_name=metadata["file_name"],
                    file_size=metadata["file_size"],
                    chunk_size=self.chunk_manager.get_chunk_size(),
                    chunk_overlap=self.chunk_manager.get_chunk_overlap(),
                    file_title=metadata["file_name"],
                )
                processed_file_ids.add(file_id)
                LogUtils.log_info(f"文档 {metadata['file_name']} 信息已添加到知识库")
            else:
                LogUtils.log_info(f"文档 {metadata['file_name']} 已存在，跳过添加")

        LogUtils.log_info(
            f"知识库加载成功，共添加 {len(processed_file_ids)} 个唯一文档"
        )

    def retrieve_chunk(
        self,
        query: str,
        rag_config: RagFrontendConfig = None,
        filters: MetadataFilters = None,
    ) -> Tuple[List[NodeWithScore], str]:
        LogUtils.log_info(f"retrieve_chunk: {query}")

        start_time = time.time()
        # 2. retrieve
        retrieve_nodes = self.retriever_manager.retrieve_chunk(
            query=query, filters=filters, rag_config=rag_config
        )
        retrieve_elapsed_time = str(round(time.time() - start_time, 2))
        LogUtils.log_info(f"retrieve_elapsed_time :{retrieve_elapsed_time} seconds")

        langfuse_callback_handler.flush()

        return retrieve_nodes, retrieve_elapsed_time

    async def aretrieve_chunk(
        self,
        query: str,
        rag_config: RagFrontendConfig = None,
        filters: MetadataFilters = None,
    ) -> Tuple[List[NodeWithScore], str]:
        return self.retrieve_chunk(query, rag_config, filters)

    def rerank_chunks(
        self,
        origin_query: str,
        retrieve_nodes: list[NodeWithScore],
        rag_config: RagFrontendConfig = None,
    ) -> Tuple[List[NodeWithScore], str]:
        start_time = time.time()

        rerank_nodes = self.rerank_manager.rerank(
            origin_query, retrieve_nodes, rag_config
        )

        rerank_elapsed_time = str(round(time.time() - start_time, 2))
        LogUtils.log_info(f"rerank_elapsed_time: {rerank_elapsed_time} seconds")

        langfuse_callback_handler.flush()

        return rerank_nodes, rerank_elapsed_time

    async def arerank_chunks(
        self,
        origin_query: str,
        retrieve_nodes: list[NodeWithScore],
        rag_config: RagFrontendConfig = None,
    ) -> Tuple[List[NodeWithScore], str]:
        return self.rerank_chunks(origin_query, retrieve_nodes, rag_config)

    def generate_image_nodes_response(
        self, query: str, image_nodes: list[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], str]:
        start_time = time.time()

        reply = self.image_qa_manager.generate_image_node_answer(query, image_nodes)

        iamge_qa_elapsed_time = str(round(time.time() - start_time, 2))
        LogUtils.log_info(f"iamge_qa_elapsed_time : {iamge_qa_elapsed_time} seconds")
        LogUtils.log_info(f"generate_image_nodes_response:\n {reply}")
        if reply:
            for image_node in image_nodes:
                image_node.node.set_content(reply)
        langfuse_callback_handler.flush()
        return image_nodes, iamge_qa_elapsed_time

    async def agenerate_image_nodes_response(
        self, query: str, image_nodes: list[NodeWithScore]
    ) -> Tuple[List[NodeWithScore], str]:
        return self.generate_image_nodes_response(query, image_nodes)

    def generate_chat_stream_response(self, query: str, nodes: list[NodeWithScore]):
        reply = self.generate_manager.generate_stream_response(query, nodes)
        # langfuse_callback_handler.flush()
        return reply
