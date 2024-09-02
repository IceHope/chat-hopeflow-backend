import os

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from path_config import PATH_PROJECT_IMAGES

from factory.llm.base_lllm_factory import BaseLLMFactory


class OpenaiLlmFactory(BaseLLMFactory):
    def get_default_mode_name(self) -> str:
        return "gpt-3.5-turbo"

    def get_llm(self, mode_name: str = None) -> BaseChatModel:
        # !pip install --upgrade langchain-openai
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.7,
            model=self.get_default_mode_name() if mode_name is None else mode_name,
        )


def perform_extract_image(image_url: str):
    llm = OpenaiLlmFactory().get_llm("gpt-4o-mini")
    inputs = [
        {"role": "user",
         "content": [
             {"type": "text", "text": "这张图片描述了什么"},
             {"type": "image_url", "image_url": {"url": image_url}},
         ]},
    ]

    response = llm.invoke(inputs)
    print("response: ", response.content)
    print("tokens: ", response.usage_metadata)


def get_image_net_url(image_path: str):
    from controller.utils.tencent_cos import get_tencent_cloud_url
    return get_tencent_cloud_url(image_path)


def get_image_base64_url(image_path: str, quality=85):
    import base64
    # 打开图片文件
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    print("img_base64 len =", len(img_base64))
    return f"data:image/jpeg;base64,{img_base64}"


def test_image_tokens():
    image_path = PATH_PROJECT_IMAGES + "/gradient.png"

    print("net_url tokens :")
    image_net_url = get_image_net_url(image_path)
    perform_extract_image(image_net_url)

    print("base64 tokens :")
    image_base64_url = get_image_base64_url(image_path)
    perform_extract_image(image_base64_url)


def generate_follow_questions():
    llm = OpenaiLlmFactory().get_llm("gpt-3.5-turbo")
    str = """
    你是一个善于引导用户思维的专家,请根据用户的提问,和对应的回答,提出{number}个相关的引导问题
    只回复最重要的引导问题,不要回复其他内容,问题要简洁,每个问题不能超过15个字符
    user_ask:{ask}
    ai_answer:{ai_answer}
    helper_questions:[your reply]
    
    例如:
    user_ask:中国有多大
    ai_answer:中国的总面积大约是960万平方公里，这使它成为世界上面积第三大的国家，仅次于俄罗斯和加拿大。中国的地理范围非常广泛，从东部的沿海地区到西部的高原和山脉，拥有丰富多样的地形和气候条件
    helper_questions
    [Q]中国的地形地貌有哪儿些特点
    [Q]中国的气候类型是怎么分布的
    [Q]中国的人口分布有哪儿些特点
    """
    ask = "中国有多大"
    ai_answer = "中国是世界上面积第三大的国家，仅次于俄罗斯和加拿大。中国的总面积大约是960万平方公里。这个数字包括了陆地面积和内陆水域，但不包括海洋领土。中国的地理范围非常广泛，从东部的沿海地区到西部的高原和山脉，地形多样，气候条件也各不相同。"

    prompt = ChatPromptTemplate.from_template(str)
    chain = prompt | llm | StrOutputParser()
    rest = chain.invoke({"number": 3, "ask": ask, "ai_answer": ai_answer})
    print(rest)


if __name__ == "__main__":
    # test_image_tokens()
    # llm = OpenaiLlmFactory().get_llm("gpt-3.5-turbo")
    # stream_response = llm.stream("你是谁")
    # for chunk in stream_response:
    #     print(chunk.content, end="")
    #
    # print(chunk)
    generate_follow_questions()
