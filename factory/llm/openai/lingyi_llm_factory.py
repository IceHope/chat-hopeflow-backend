import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class LingyiLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "yi-large-turbo"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("LINGYI_API_KEY"),
            base_url=os.getenv("LINGYI_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


if __name__ == '__main__':
    messages = [
        {"role": "user", "content": "介绍下你自己,版本号,生产商"},
        {"role": "assistant", "content": "我是一个AI助手,我的版本号是1.0.6,我是由Lingyi团队开发的"},
        {"role": "user", "content": "我的名字是kimi"},
        {"role": "assistant", "content": "你好kimi"},
        {"role": "user", "content": "你是谁"},
    ]

    from factory.mode_names_versions import LINGYI_NAMES

    llm = LingyiLlmFactory().get_llm(LINGYI_NAMES[0][0])

    stream_response = llm.stream(messages)
    for chunk in stream_response:
        print(chunk.content, end="")
