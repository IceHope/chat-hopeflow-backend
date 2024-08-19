import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class DashscopeLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "qwen-plus"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name
        )


if __name__ == "__main__":
    image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    llm = DashscopeLlmFactory().get_llm("qwen-vl-plus")
    inputs = [
        {"role": "user",
         "content": [
             {"type": "text", "text": "这张图片描述了什么"},
             {"type": "image_url", "image_url": {"url": image_url}},
         ]},
    ]
    # stream_response = llm.stream(inputs)
    # for chunk in stream_response:
    #     print(chunk.content)
    response = llm.invoke(inputs)
    print(response)
