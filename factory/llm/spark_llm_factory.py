import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class SparkLLMFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        pass

    # 默认是MAX

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        # pip install websocket-client
        from langchain_community.chat_models import ChatSparkLLM
        return ChatSparkLLM(
            spark_app_id=os.getenv("IFLYTEK_SPARK_APP_ID"),
            spark_api_key=os.getenv("IFLYTEK_SPARK_API_KEY"),
            spark_api_secret=os.getenv("IFLYTEK_SPARK_API_SECRET"),
        )


if __name__ == "__main__":
    llm = SparkLLMFactory().get_llm()
    stream_response = llm.stream("1000字的散文")
    for chunk in stream_response:
        print(chunk.content, end="|")
    # response = llm.invoke("介绍下你自己")
    # print(response)
