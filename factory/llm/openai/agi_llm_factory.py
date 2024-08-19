import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class AgiLLMFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        pass

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("OPEN_AGI_API_KEY"),
            base_url=os.getenv("OPEN_AGI_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


def _test_llm():
    llm = AgiLLMFactory().get_llm("claude-3-5-sonnet-20240620")
    stream_response = llm.stream("你是谁")
    for chunk in stream_response:
        print(chunk.content, end="")


def _test_img():
    # image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    # image_url = "https://icehope-1326453681.cos.ap-beijing.myqcloud.com/2024-08-02-23-28-39.png"
    image_url = "https://img1.baidu.com/it/u=1369931113,3388870256&fm=253&app=138&size=w931&n=0&f=JPEG&fmt=auto?sec=1703696400&t=f3028c7a1dca43a080aeb8239f09cc2f"
    llm = AgiLLMFactory().get_llm("claude-3-5-sonnet-20240620")
    inputs = [
        {"role": "user",
         "content": [
             {"type": "text", "text": "这张图片描述了什么"},
             {"type": "image_url", "image_url": {"url": image_url}},
         ]},
    ]
    # stream_response = llm.stream(inputs)
    # for chunk in stream_response:
    #     print(chunk.content, end="")
    response = llm.invoke(inputs)
    print(response)


if __name__ == '__main__':
    # _test_llm()
    _test_img()