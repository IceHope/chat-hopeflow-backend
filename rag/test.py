import pprint

from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

from rag.managers.reader_manager import ReaderManager
from rag.rag_base_manager import RagBaseManager


def _manual_load_pdf(dir):
    reader_manager = ReaderManager()

    reader_manager.manual_load_pdf(dir)


def load(dir):
    hope_manager = RagBaseManager()
    hope_manager.manual_load_file_dir(dir)


def main():
    file_id = "f29dd517-301e-5620-adc8-2ef3a737fd50"
    filters = MetadataFilters(filters=[ExactMatchFilter(key="file_id", value=file_id)])
    hope_manager = RagBaseManager()
    while True:
        query = input("请输入查询内容：")
        if query == "exit":
            break
        nodes = hope_manager.retrieve_chunk(query)
        print("-----------------nodes-------------------------")
        for node in nodes:
            pprint.pprint(node)
        print("--------------------reply----------------------")
        reply = hope_manager.generate_chat_stream_response(query, nodes)
        for text in reply.response_gen:
            print(text, end="|")
            # pass
        # print("reply: ", reply)


if __name__ == "__main__":
    main()
    # load()
    # dir = "F:/AiData/test/test1"
    # _manual_load_pdf(dir)
    # load(dir)
    # from llama_index.llms.groq import Groq
    #
    # Settings.llm = Groq(
    #     model="llama-3.1-8b-instant",
    #     api_key=os.getenv("GROQ_API_KEY"),
    #     api_base=os.getenv("GROQ_BASE_URL"),
    #     temperature=0.7,
    # )
    #
    # reply = Settings.llm.complete("你是谁")
    # print(reply)
