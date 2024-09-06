import os
import pprint

from dotenv import load_dotenv
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, PromptTemplate, StorageContext
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader

from rag.prompt.prompt import RAG_QUERY_PROMPT
from models.rerank.reranker import RagReranker
from utils.log_utils import LogUtils

load_dotenv()
langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])

CHROMA_COLLECTION_NAME = "BM25_chroma"


def get_create_chroma_client(collection_name: str):
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    # 向量数据库
    chroma_client = chromadb.PersistentClient()
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def init():
    from llama_index.llms.groq import Groq

    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        api_base=os.getenv("GROQ_BASE_URL"),
        temperature=0.7,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="F:/HuggingFace/Embedding/bge-small-en-v1.5"
    )


# 数据加载流程,实际业务因该是早就加载完数据了,后面只是单纯的查询功能
def load_data():
    # path_dir = "F:/AiData/paul_graham_essay.txt"
    path_dir = "F:/AGI/data/llama2.pdf"

    documents = SimpleDirectoryReader(
        input_files=[path_dir],
        recursive=False,
        # required_exts=[".pdf", ".txt", ".md"],
        file_extractor={".pdf": PyMuPDFReader()},
    ).load_data()

    node_parser = SentenceSplitter(chunk_size=512)
    nodes = node_parser.get_nodes_from_documents(documents)

    LogUtils.log_info(f"load data success, total {len(nodes)} nodes")

    # initialize a docstore to store nodes
    # also available are mongodb, redis, postgres, etc for docstores
    from llama_index.core.storage.docstore import SimpleDocumentStore

    docstore = SimpleDocumentStore.from_persist_path("./docstore.json")
    # docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # vector_store = get_create_milvus_client(overwrite=False)
    vector_store = get_create_chroma_client(CHROMA_COLLECTION_NAME)

    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )

    storage_context.docstore.persist("./docstore.json")

    # 索引  灌库
    # TODO 索引类型是什么?怎么自定义

    index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, show_progress=True
    )

    langfuse_callback_handler.flush()


# 1. index 数据已经在本地数据库中了,只能从本地数据库获取加载index
def get_index_from_db():
    # vector_store = get_create_milvus_client(overwrite=False)

    vector_store = get_create_chroma_client(CHROMA_COLLECTION_NAME)

    docstore = SimpleDocumentStore.from_persist_path("./docstore.json")

    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=3,
        # Optional: We can pass in the stemmer and set the language for stopwords
        # This is important for removing stopwords and stemming the query + text
        # The default is english for both
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )

    # 从本地数据库加载index
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    return index, bm25_retriever


# 2. retriever
def get_fusion_retriever(index: VectorStoreIndex):
    """ RAG Fusion 相当于重试按钮,重试了num_queries-1次
    1.查询大模型返回num_queries-1个相似问题
    2.用原始问题和相似问题生成[query]集合
    3.用[query]集合匹配向量数据库,返回num_queries个查询集合,每个集合有similarity_top_k条结果
    4.用RRF重排序
    """

    # index初始化不行,没有nodes数据,Llamaindex从数据库加载的index没有nodes数据
    # 如果想看效果,只能是load_data的时候把nodes保存在本地
    # bm25_retriever = BM25Retriever.from_defaults(index=index)
    # loaded_bm25_retriever = BM25Retriever.from_persist_dir("./bm25_retriever")

    # query->querys之后,仍然调用base_retriever
    filters = MetadataFilters(
        filters=[MetadataFilter(key="file_name", value="国家人工智能产业综合标准化体系建设指南（2024版）.pdf")]
    )

    base_retriever = index.as_retriever(similarity_top_k=6, filters=filters)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[
            base_retriever,
            # loaded_bm25_retriever
        ],
        mode=FUSION_MODES.RECIPROCAL_RANK,  # RRF
        num_queries=1,  # 生成 query数 ,包含了原始查询
        use_async=True,
        # query_gen_prompt="...",  # 可以自定义 query 生成的 prompt 模板,比如中英文混合的query
    )
    return fusion_retriever


# 3. query大模型
def get_query_engine(retriever: QueryFusionRetriever):
    # Rerank 很重要,优化了查询,去掉相关性最差的,节省查询token
    reranker = RagReranker.get_cross_encoder(top_n=3)

    # 查询大模型
    query_engine = RetrieverQueryEngine(
        retriever,
        node_postprocessors=[reranker],
        # response_synthesizer=get_response_synthesizer(streaming=True),
    )

    # 更新 prompt; 定义只能用中文回答
    rag_prompt_tmpl = PromptTemplate(RAG_QUERY_PROMPT)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": rag_prompt_tmpl})
    langfuse_callback_handler.flush()
    return query_engine


def main():
    # load_data()

    index = get_index_from_db()

    retriever = get_fusion_retriever(index)

    query_engine = get_query_engine(retriever)

    while True:
        question = input("\nUser:")
        if question.strip() == "":
            break

        response = query_engine.query(question)
        # for text in streaming_response.response_gen:
        #     print(text, end="")
        #     pass
        # langfuse_callback_handler.flush()
        pprint.pprint(response)


def demo_retriever():
    index = get_index_from_db()

    fusion_retriever = get_fusion_retriever(index)

    question = "GPT4什么时候发布的"
    # question = "LLama2什么时候发布的"
    nodes = fusion_retriever.retrieve(question)
    for i, node in enumerate(nodes):
        print("\n\n", i, "---\n", node.node.text)
    langfuse_callback_handler.flush()


if __name__ == "__main__":
    init()
    # load_data()
    index, bm25_retriever = get_index_from_db()
    while True:
        question = input("\nUser:")
        if question.strip() == "":
            break
        # "What happened at Viaweb and Interleaf?"
        nodes = bm25_retriever.retrieve(question)
        for i, node in enumerate(nodes):
            print("\n\n", i, "score:", node.score, "---\n", node.node.text)
        # main()
        # demo_retriever()
        # decomposition("法国和英国去年的GDP谁更大")
