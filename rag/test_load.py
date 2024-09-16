import pprint

from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

from dao.knowledge_dao import KnowledgeDao
from rag.managers.reader_manager import ReaderManager
from rag.rag_base_manager import RagBaseManager


def question():
    file_id = "4383d71c-f298-5abc-92e5-a3903efb4b2d"
    filters = MetadataFilters(filters=[ExactMatchFilter(key="file_id", value=file_id)])
    hope_manager = RagBaseManager()
    while True:
        query = input("请输入查询内容：")
        if query == "exit":
            break
        nodes = hope_manager.retrieve_chunk(query)[0]
        print("-----------------nodes-------------------------")
        for node in nodes:
            pprint.pprint(node)
        print("--------------------reply----------------------")
        reply = hope_manager.generate_chat_stream_response(query, nodes)
        for text in reply.response_gen:
            print(text, end="")
            # pass
        # print("reply: ", reply)


def preload(dir):
    reader_manager = ReaderManager()
    reader_manager.manual_load_pdf(dir)


def load_data(dir):
    hope_manager = RagBaseManager()
    hope_manager.manual_load_file_dir(dir)


def add_konwledge():
    knowledge_dao = KnowledgeDao()
    user_name = "admin_test"
    file_id = "48d6d28c-f740-5b9c-9025-6904e5cf039d"
    file_path = "F:\\AiData\\stable\\国家人工智能产业综合标准化体系建设指南_2024.pdf"
    file_name = "国家人工智能产业综合标准化体系建设指南_2024.pdf"
    file_size = 718686
    chunk_size = 800
    chunk_overlap = 200
    knowledge_dao.add_new_knowledge(
        user_name=user_name,
        file_id=file_id,
        file_path=file_path,
        file_name=file_name,
        file_size=file_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        file_title=file_name,
    )


if __name__ == "__main__":
    # dir = "F:/AiData/stable"
    # preload(dir)
    # load_data(dir)
    # question()
    add_konwledge()
