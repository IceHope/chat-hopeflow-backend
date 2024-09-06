import os

from langchain_core.language_models import BaseChatModel

from models.llm.langchain.base_llm_factory import BaseLLMFactory


class QianfanLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "ERNIE-Speed-Pro-128K"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        # pip install --upgrade langchain_community
        # pip install --upgrade qianfan
        from langchain_community.chat_models import QianfanChatEndpoint

        return QianfanChatEndpoint(
            qianfan_ak=os.getenv("ERNIE_CLIENT_ID"),
            qianfan_sk=os.getenv("ERNIE_CLIENT_SECRET"),

            # model="ERNIE-4.0-8K-Latest",
            model="ernie-4.0-turbo-8k",
            # endpoint="ernie-4.0-turbo-8k",
        )


if __name__ == "__main__":

    llm = QianfanLlmFactory().get_llm("ERNIE-Speed-Pro-128K")
    stream_response = llm.stream("介绍下你自己")
    for chunk in stream_response:
        print(chunk.content, end="|")
