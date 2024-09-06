from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.factory.llm_factory import LLMFactory
from models.model_type import LLMType
from utils.log_utils import LogUtils

SUGGEST_QUESTION_PROMPT = """
    你是一个善于引导用户思维的专家,请根据用户的提问,和对应的回答,提出{number}个相关的引导问题
    只回复最重要的引导问题,不要回复其他内容,问题要简洁,
    每个问题不能超过15个字符,以[Q]开头,以?结尾
    user_ask:{ask}
    ai_answer:{ai_answer}
    helper_questions:
    [Q]

    下面是一个具体的例子:
    user_ask:中国有多大
    ai_answer:中国的总面积大约是960万平方公里，这使它成为世界上面积第三大的国家，仅次于俄罗斯和加拿大。中国的地理范围非常广泛，从东部的沿海地区到西部的高原和山脉，拥有丰富多样的地形和气候条件
    helper_questions:
    [Q]中国的地形地貌有哪儿些特点?
    [Q]中国的气候类型是怎么分布的?
    [Q]中国的人口分布有哪儿些特点?
"""


follow_question_llm = LLMFactory.get_llm(LLMType.ZHIPU, "GLM-4-Flash")


def generate_follow_questions(ask: str, reply: str):
    suggest_prompt = ChatPromptTemplate.from_template(SUGGEST_QUESTION_PROMPT)
    suggest_chain = suggest_prompt | follow_question_llm | StrOutputParser()
    follow_questions = suggest_chain.invoke(
        {"number": 3, "ask": ask, "ai_answer": reply}
    )
    LogUtils.log_info(f"follow_questions: {follow_questions}")
    return follow_questions
