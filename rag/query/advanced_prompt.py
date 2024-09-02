SUB_QUESTION_PROMPT = """"
你是一位精通问题分析和拆解的专家。你的任务是判断用户的问题是否需要查询多个数据源才能得到完整答案。

请按以下规则处理用户问题:

1. 如果问题需要对比或涉及多个方面,请将其拆解为多个单一问题,以列表形式返回。
   每个子问题应该只针对一个具体方面或数据源。

2. 如果问题是单一查询,无需拆解,请直接返回原问题。

3. 拆解时,保持问题的原意,使用简洁明了的语言。

示例1 - 需要拆解:
用户问题: 比较2010年和2020年中国、美国的GDP。
拆解结果:
2010年中国的GDP是多少?
2010年美国的GDP是多少?
2020年中国的GDP是多少?
2020年美国的GDP是多少?


示例2 - 需要拆解:
用户问题: 中国在2023年的人口相比2022年的增长了多少
拆解结果:
中国在2022年的人口数量是多少?
中国在2023年的人口数量是多少?

示例3 - 无需拆解:
用户问题: 2023年诺贝尔物理学奖获得者是谁?
结果: 2023年诺贝尔物理学奖获得者是谁?

示例4 - 无需拆解:
用户问题: 中国的首都在哪儿
结果: 中国的首都在哪儿

现在,请分析以下用户问题,并按上述规则处理:

用户问题: {query_str}

你的回答:

"""

METADATA_PROMPT = """
你是一位精通多领域技术文档分类的专家分析师。请根据用户的问题,准确判断其所属类别,并返回相应的唯一标识符。类别定义如下:

source_langchain: LangChain框架相关的技术文档和使用方法
source_llamaindex: LlamaIndex库的文档、API和应用
source_python: Python编程语言的官方文档、库使用和最佳实践
source_openai: OpenAI公司的API文档、模型说明和开发指南
source_all: 不属于上述特定类别的通用问题,需要综合知识库

请注意:
1. 只能返回以上5个类别中的一个,不可返回多个或超出定义范围的类别
2. 分析时请考虑问题的核心主题、使用的技术术语和上下文
3. 如有疑虑,优先选择最相关的特定类别,而非source_all

示例:
用户: "LlamaIndex如何实现文档分割?"
回复: source_llamaindex

用户: "Python中如何使用async/await?"
回复: source_python

用户: "请解释量子计算的基本原理"
回复: source_all

用户最新提问: {query_str}
你的分类回复:

"""

STEP_BACK_PROMPT = """
You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
Examples:
Original query: What are the impacts of climate change on the environment?
Step-back query:What are the general effects of climate change?

Belows is user query:

Original query: {original_query}
Step-back query:

"""

QUERY_REWRITE_TEMPLATE = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
    Examples:
    Original query: What are the impacts of climate change on the environment?
    Rewritten query:What are the specific effects of climate change on various ecosystems, including changes in temperature, precipitation patterns, sea levels, and biodiversity?

    Belows is user query:
    Original query: {original_query}

    Rewritten query:"""


# def generate_rewrite():
#     from llama_index.core import PromptTemplate
#     query = "langchain怎么使用"
#     prompt = PromptTemplate(query_rewrite_template).format(original_query=query)
#
#     questions = Settings.llm.complete(prompt).text
#     print(questions)


def generate_hyde_query():
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.core.indices.query.query_transform.base import (
        HyDEQueryTransform,
    )
    from llama_index.core.query_engine import TransformQueryEngine

    # load documents, build index
    documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
    index = VectorStoreIndex(documents)

    # run query with HyDE query transform
    query_str = "what did paul graham do after going to RISD"
    hyde = HyDEQueryTransform(include_original=True)
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(query_engine, query_transform=hyde)
    response = query_engine.query(query_str)
    print(response)


def main():
    # generate_sub_questions()
    # generate_matadata()
    # generate_step_back()
    # generate_rewrite()
    generate_hyde_query()


if __name__ == "__main__":
    main()
