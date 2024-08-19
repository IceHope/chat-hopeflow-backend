from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class HunyuanLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        pass

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_community.chat_models import ChatHunyuan
        return ChatHunyuan(
            model="hunyuan-vision"
        )


if __name__ == "__main__":
    llm = HunyuanLlmFactory().get_llm()
    stream_response = llm.stream("介绍下你自己,版本号,生产商")
    for chunk in stream_response:
        print(chunk, end="")
    # response = llm.invoke("介绍下你自己")
    # print(response)
