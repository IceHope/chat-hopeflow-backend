import pprint

from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

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


if __name__ == "__main__":
    dir = "F:/AiData/stable"
    # preload(dir)
    load_data(dir)
    # question()
