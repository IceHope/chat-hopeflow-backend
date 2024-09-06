import time
from typing import List

from utils.log_utils import LogUtils
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.factory.llm_factory import LLMFactory
from models.model_type import LLMType
from utils.log_utils import LogUtils

CONTEXT_PARSE_PROMPT = """
你是一个擅长从对话历史中,推断用户真实意图的智能解析助手，你的任务是根据聊天记录与用户最新的问题，
推断并补全用户想要问的问题。请确保你给出的补全问题能够准确反映用户的意图。
请直接回复问题,不要去尝试解答
回答的内容不能包含比如[[这些,那些,他们]],这些容易混淆的指代性词语,必须转换成具体的名称含义

[例1
chat_history:
<< {{ "role": "user", "content": "中国的面积有多大？" }}
{{ "role": "assistant", "content": "中国的总面积大约是960万平方公里，仅次于俄罗斯和加拿大" }} >>
user_ask: 美国呢？
your answer: 美国的面积有多大？

[例2]
chat_history:
<< {{ "role": "user", "content": "中国的面积有多大？" }}
{{ "role": "assistant", "content": "中国的总面积大约是960万平方公里，仅次于俄罗斯和加拿大" }} >>
user_ask: 人口呢？
your answer: 中国的人口是多少？

[例3]
chat_history:
<< {{ "role": "user", "content": "请推荐一些适合初学者的编程书籍。" }}
{{ "role": "assistant", "content": "适合初学者的编程书籍包括：《Python编程：从入门到实践》、Head First系列等" }} >>
{{ "role": "user", "content": "这些书可以在网上买吗？" }}
{{ "role": "assistant", "content": "是的，这些书可以在各大在线书店购买，如亚马逊、当当等。" }} >>
user_ask: 那Java呢？
your answer: 请推荐一些适合初学者的Java编程书籍。

现在，请根据以下聊天记录，补全用户的问题：
chat_history:
<<{context}>>
user_ask:{ask}
your answer:
"""


class QueryManager:
    def __init__(self):
        LogUtils.log_info("QueryManager initialized")
        self.parse_context_llm = LLMFactory.get_llm(LLMType.ZHIPU, "GLM-4-Flash")

    def query_rewrite(self, query) -> List[str]:
        # TODO: Implement query rewrite logic
        return [query]

    def parse_context_question(self, origin_query: str, context: str):
        start_time = time.time()
        if context is None:
            return origin_query, "0"

        rag_parse_prompt = ChatPromptTemplate.from_template(CONTEXT_PARSE_PROMPT)
        parse_chain = rag_parse_prompt | self.parse_context_llm | StrOutputParser()
        parse_question = parse_chain.invoke({"ask": origin_query, "context": context})

        parse_elapsed_time = str(round(time.time() - start_time, 2))
        LogUtils.log_info(f"parse_elapsed_time:{parse_elapsed_time} seconds")
        LogUtils.log_info(f"parse_question: {parse_question}")
        return parse_question, parse_elapsed_time


if __name__ == "__main__":
    # query_manager = QueryManager()
    rest = CONTEXT_PARSE_PROMPT.format(context="test", ask="test")
    print(rest)
