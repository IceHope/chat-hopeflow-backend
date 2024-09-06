from models.api_key_config import ApiKeyUrlConfig, get_agi_type_config
from models.llm.langchain.openai.base_openai_llm_factory import BaseOpenaiTypeFactory


class BaseAgiLLMFactory(BaseOpenaiTypeFactory):
    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        return get_agi_type_config()


def _test_llm():
    llm = BaseAgiLLMFactory().get_llm("claude-3-5-sonnet-20240620")
    stream_response = llm.stream("你是谁")
    for chunk in stream_response:
        print(chunk.content, end="")


def _test_img():
    # image_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
    # image_url = "https://icehope-1326453681.cos.ap-beijing.myqcloud.com/2024-08-02-23-28-39.png"
    image_url = "https://img1.baidu.com/it/u=1369931113,3388870256&fm=253&app=138&size=w931&n=0&f=JPEG&fmt=auto?sec=1703696400&t=f3028c7a1dca43a080aeb8239f09cc2f"
    llm = BaseAgiLLMFactory().get_llm("gemini-1.5-flash")
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
