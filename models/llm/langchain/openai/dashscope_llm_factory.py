from models.api_key_config import ApiKeyUrlConfig, get_dashscope_config
from models.llm.langchain.openai.base_openai_llm_factory import BaseOpenaiTypeFactory


class DashscopeLlmFactory(BaseOpenaiTypeFactory):

    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        return get_dashscope_config()


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
