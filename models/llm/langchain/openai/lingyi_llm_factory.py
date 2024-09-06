from models.api_key_config import get_lingyi_config, ApiKeyUrlConfig

from models.llm.langchain.openai.base_openai_llm_factory import BaseOpenaiTypeFactory


class LingyiLlmFactory(BaseOpenaiTypeFactory):

    def get_api_key_url_config(self) -> ApiKeyUrlConfig:
        return get_lingyi_config()



if __name__ == '__main__':
    messages = [
        {"role": "user", "content": "介绍下你自己,版本号,生产商"},
        {"role": "assistant", "content": "我是一个AI助手,我的版本号是1.0.6,我是由Lingyi团队开发的"},
        {"role": "user", "content": "我的名字是kimi"},
        {"role": "assistant", "content": "你好kimi"},
        {"role": "user", "content": "你是谁"},
    ]

    llm = LingyiLlmFactory().get_llm()

    stream_response = llm.stream(messages)
    for chunk in stream_response:
        print(chunk.content, end="")
