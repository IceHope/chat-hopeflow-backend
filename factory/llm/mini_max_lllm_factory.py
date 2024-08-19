import os

from langchain_core.language_models import BaseChatModel

from factory.llm.base_lllm_factory import BaseLLMFactory


class MinimaxLlmFactory(BaseLLMFactory):

    def get_default_mode_name(self) -> str:
        return "abab6.5s-chat"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        from langchain_community.chat_models import MiniMaxChat

        return MiniMaxChat(
            minimax_group_id=os.getenv("MINIMAX_GROUP_ID"),
            minimax_api_key=os.getenv("MINIMAX_API_KEY"),
            minimax_api_host=os.getenv("MINIMAX_API_HOST"),
            model=self.get_default_mode_name() if mode_name is None else mode_name
        )


if __name__ == "__main__":
    from factory.mode_names_versions import MINIMAX_NAMES

    llm = MinimaxLlmFactory().get_llm(MINIMAX_NAMES[-1][0])
    stream_response = llm.stream("介绍下你自己")
    for chunk in stream_response:
        print(chunk.content, end="|")
