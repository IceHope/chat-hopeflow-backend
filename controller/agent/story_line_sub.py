import asyncio
import operator
import os
import pprint
from datetime import datetime
from typing import TypedDict, Annotated

from fastapi import APIRouter
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket

from models.factory.llm_factory import LLMFactory
from models.model_type import LLMType

translate_human_router = APIRouter()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "agent-translate" + datetime.now().strftime(
    "%d/%m/%Y %H:%M:%S"
)


class StoryLineDetail(BaseModel):
    detail_effect: str = Field(
        alias="本段故事作用",
        description="本段故事作用,描述本段故事在整体结构中发挥的作用",
    )
    key_plot: str = Field(
        alias="关键情节",
        description="关键情节,按时序描述本段故事中的关键情节，以及情节中的关键细节",
    )
    key_role: str = Field(
        alias="涉及关键人物", description="涉及关键人物,给出本段故事中涉及的关键人物名"
    )

    class Config:
        populate_by_name = True


class StoryLine(BaseModel):
    plot_struct_type: str = Field(
        alias="情节结构类型",
        description="情节结构类型,基于常见的故事、小说、剧作创作方法，输出你将要使用的剧情结构类型名称",
    )
    polt_struct_character: str = Field(
        alias="情节结构特点",
        description="情节结构特点,阐述{情节结构类型}的剧情结构手法、特点",
    )
    detailed_story_line: list[StoryLineDetail] = Field(
        alias="故事线详细创作", description="故事线详细创作的顺序列表"
    )

    class Config:
        populate_by_name = True


class StoryState(TypedDict):
    llm: BaseChatModel
    web_socket: WebSocket
    idea: str
    background: str
    storyline: StoryLine
    background_human_flag: bool
    storyline_human_flag: bool
    storyline_detail_human_flag: bool


class BackgroundState(StoryState):
    background_human_feedback: str


class StorylineState(StoryState):
    storyline_human_feedback: str


class StorylineDetailState(StoryState):
    storyline_detail_human_feedback: str
    storyline_details: list[StoryLineDetail]
    current_detail_index: int
    stories: Annotated[list[str], operator.add]


async def generate_background(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    idea = state.get("idea")

    background_prompt = """请根据故事灵感[{idea}]创作故事的世界信息和背景故事，其中：
世界信息需要包括世界的主要国家或地区分布，不同国家或地区的环境描写，科技水平，信仰情况等
世界背景故事需要以时间线的形式描述世界的主要历史沿革，国家或地区之间的重大事件及带来的影响变化等
输出格式如下
{{
    "世界名称": "str",
    "主要国家或地区": [{{
        "名称": "str",
        "关键信息": "str",
    }}],
    "世界背景故事": ["str"],
}}
"""

    prompt = background_prompt.format(
        idea=idea
    )

    print_msg = "# " + idea + "\n" + "## 世界观背景故事\n"
    print(print_msg)
    # await web_socket.send_text(print_msg)

    stream_response = llm.stream(prompt)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        # await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        # await web_socket.send_text(str(chunk.content))

        response += chunk.content

    return {
        "background": response,
    }


async def background_human_feedback_node(state):
    web_socket = state.get("web_socket")
    # 等待前端的反馈
    # await web_socket.send_text(
    #     "\n #### 生成的内容是否满意，如果满意请直接回复Y,如果不满意请发送修改意见"
    # )
    # await web_socket.send_text(CHAT_STREAM_SERVE_DONE)
    #
    # chat_data = await web_socket.receive_text()
    # print("\nHuman feedback:", chat_data)
    #
    # msg = StoryLineAgentSchema(**json.loads(chat_data))
    msg = input("生成的内容是否满意，如果满意请直接回复Y,如果不满意请发送修改意见\n")
    if msg.upper().startswith("Y"):
        return {
            "background_human_feedback": "",
        }

    return {
        "background_human_feedback": msg,
    }


async def modify_background_node(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    idea = state.get("idea")
    background = state.get("background")
    background_human_feedback = state.get("background_human_feedback")

    modify_prompt = """已知故事灵感[{idea}]和已经创作了故事的世界信息和背景故事
            请根据用户反馈进行修改,格式跟原来保持一致,只回答修改后的信息
            已经创作的信息:[{background}]
            用户修改意见:[{background_human_feedback}]
            你的回答:
            """
    prompt = modify_prompt.format(
        idea=idea,
        background=background,
        background_human_feedback=background_human_feedback,
    )

    stream_response = llm.stream(prompt)

    print_msg = "# " + idea + "\n" + "## 修改后的世界观背景故事\n"
    print(print_msg)
    # await web_socket.send_text(print_msg)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        # await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        # await web_socket.send_text(str(chunk.content))

        response += chunk.content

    return {
        "background": response,
    }


async def generate_finish_node(state):
    # web_socket = state.get("web_socket")
    # await web_socket.send_text(CHAT_STREAM_SERVE_DONE)
    print("\n--generate_finish_node--\n")
    return {"background_human_flag": state.get("background_human_flag")}


NODE_GENERATE_BACKGROUND = "generate_background_node"
NODE_MODIFY_BACKGROUND = "modify_background_node"
NODE_BACKGROUND_HUMAN_FEEDBACK = "background_human_feedback_node"
NODE_GENERATE_FINISH = "generate_finish_node"


def background_router(state):
    background_human_flag = state.get("background_human_flag")
    if background_human_flag:
        return NODE_BACKGROUND_HUMAN_FEEDBACK

    return NODE_GENERATE_FINISH


def background_human_router(state):
    background_human_feedback = state.get("background_human_feedback")
    if background_human_feedback:
        return NODE_MODIFY_BACKGROUND
    return NODE_GENERATE_FINISH


def get_background_graph():
    workflow = StateGraph(BackgroundState)
    workflow.add_node(NODE_GENERATE_BACKGROUND, generate_background)
    workflow.add_node(NODE_MODIFY_BACKGROUND, modify_background_node)
    workflow.add_node(NODE_BACKGROUND_HUMAN_FEEDBACK, background_human_feedback_node)
    workflow.add_node(NODE_GENERATE_FINISH, generate_finish_node)

    workflow.set_entry_point(NODE_GENERATE_BACKGROUND)
    workflow.set_finish_point(NODE_GENERATE_FINISH)

    workflow.add_conditional_edges(
        NODE_GENERATE_BACKGROUND,
        background_router,
        {
            NODE_BACKGROUND_HUMAN_FEEDBACK: NODE_BACKGROUND_HUMAN_FEEDBACK,
            NODE_GENERATE_FINISH: NODE_GENERATE_FINISH
        }
    )
    workflow.add_conditional_edges(
        NODE_BACKGROUND_HUMAN_FEEDBACK,
        background_human_router,
        {
            NODE_MODIFY_BACKGROUND: NODE_MODIFY_BACKGROUND,
            NODE_GENERATE_FINISH: NODE_GENERATE_FINISH
        }
    )
    workflow.add_edge(NODE_MODIFY_BACKGROUND, NODE_BACKGROUND_HUMAN_FEEDBACK)
    return workflow


async def generate_storyline_node(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    idea = state.get("idea")
    background = state.get("background")

    storyline_prompt = """请根据世界观背景故事[{background}]，围绕故事灵感[{idea}]，
               创作故事的关键情节线安排,中文回答
               {model_format_instructions}
               """
    storyline_pydantic_parse = PydanticOutputParser(pydantic_object=StoryLine)
    prompt = storyline_prompt.format(
        background=background,
        idea=idea,
        model_format_instructions=storyline_pydantic_parse.get_format_instructions(),
    )
    stream_response = llm.stream(prompt)

    print_msg = "# " + idea + "\n" + "## 关键情节线安排\n"
    print(print_msg)
    await web_socket.send_text(print_msg)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content

    # fix_parse = OutputFixingParser.from_llm(parser=pydantic_parse, llm=deep_seek_llm)
    storyline = storyline_pydantic_parse.parse(response)

    # print(type(storyline))
    # pprint.pprint(storyline.model_dump_json(by_alias=True))

    return {
        "storyline": storyline,
    }


async def modify_storyline_node(state):
    llm = state.get("llm")
    idea = state.get("idea")
    background = state.get("background")
    storyline = state.get("storyline")
    storyline_human_feedback = state.get("storyline_human_feedback")

    prompt = f"""
       目前已知的信息有:
       故事灵感: {idea}
       世界观背景故事:{background}
       创作故事的关键情节线安排:{storyline.model_dump_json(by_alias=True)}
       针对情节安排,用户提出了修改意见,请根据反馈意见,结合已知的信息,对情节安排进行优化修改
       用户没有反馈的地方,不要修改,格式跟原来的保持一致
       用户的反馈意见:[{storyline_human_feedback}]
       你最终的修改:
       """
    stream_response = llm.stream(prompt)

    print_msg = "# " + idea + "\n" + "##关键情节线安排(修改后)\n"
    print(print_msg)
    # await web_socket.send_text(print_msg)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        # await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        # await web_socket.send_text(str(chunk.content))
        response += chunk.content

    return {
        "storyline": response,
    }


async def storyline_human_feedback_node(state):
    web_socket = state.get("web_socket")
    # 等待前端的反馈
    # await web_socket.send_text(
    #     "\n #### 生成的内容是否满意，如果满意请直接回复Y,如果不满意请发送修改意见"
    # )
    # await web_socket.send_text(CHAT_STREAM_SERVE_DONE)
    #
    # chat_data = await web_socket.receive_text()
    # print("\nHuman feedback:", chat_data)
    #
    # msg = StoryLineAgentSchema(**json.loads(chat_data))
    msg = input("故事情节线是否满意?如果满意请直接回复Y,如果不满意请发送修改意见\n")
    if msg.upper().startswith("Y"):
        return {
            "storyline_human_feedback": "",
        }

    return {
        "storyline_human_feedback": msg,
    }


async def generate_storyline_finish_node(state):
    # web_socket = state.get("web_socket")
    # await web_socket.send_text(CHAT_STREAM_SERVE_DONE)
    print("\n--generate_storyline_finish_node--\n")
    return {"storyline_human_flag": state.get("storyline_human_flag")}


NODE_GENERATE_STORYLINE = "generate_storyline_node"
NODE_MODIFY_STORYLINE = "modify_storyline_node"
NODE_STORYLINE_HUMAN_FEEDBACK = "storyline_human_feedback_node"
NODE_STORYLINE_GENERATE_FINISH = "generate_storyline_finish_node"


async def generate_storyline_details(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    idea = state.get("idea")
    background = state.get("background")
    storyline = state.get("storyline")

    storyline_details = state.get("storyline_details")
    current_detail_index = state.get("current_detail_index")
    stories = state.get("stories")

    last_block_content = ""
    if len(stories) > 0:
        # 在这里取上一段落的最后50个字，可根据需要修改保留的长度
        keep_length = 50
        last_block = stories[-1][(-1 * keep_length):]
        last_block_content = f'创作时需要承接[上一段落的末尾:{last_block}，确保表达的连贯性'

    current_detail = storyline_details[current_detail_index]

    storyline_detail_prompt = f"""请根据故事灵感[{idea}],世界观背景故事[{background}]，故事情节线安排[{storyline}]，
              继续创作具体的故事情节内容,{last_block_content}要求如下:
              根据本段故事的作用[{current_detail.detail_effect}]及涉及关键人物[{current_detail.key_role}]，
              将关键情节[{current_detail.key_plot}]扩写为完整的故事,
              每段故事需要尽量包括行动描写、心理活动描写和对白等细节,
              每次创作只是完整文章结构中的一部分，承担本段故事作用说明的作用任务，只需要按要求完成关键情节的描述即可，不需要考虑本段故事自身结构的完整性.
              输出的格式限定如下:
              ### {current_detail.detail_effect}\n
              [具体内容]
              """
    print_msg = "\n\n"
    # print(print_msg)
    # await web_socket.send_text(print_msg)
    stream_response = llm.stream(storyline_detail_prompt)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        # await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        # await web_socket.send_text(str(chunk.content))
        response += chunk.content

    return {
        "stories": [response],
        "current_detail_index": current_detail_index + 1
    }


async def modify_storyline_detail(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    idea = state.get("idea")
    background = state.get("background")
    storyline = state.get("storyline")

    storyline_details = state.get("storyline_details")
    current_detail_index = state.get("current_detail_index")
    stories = state.get("stories")

    current_detail = storyline_details[current_detail_index - 1]

    _detail_modify_feedback = state.get("storyline_detail_human_feedback")

    storyline_detail_modify_prompt = f"""
     已知的信息:故事灵感[{idea}],世界观背景故事[{background}]，故事情节线安排[{storyline}]，
                 
     本段故事的作用[{current_detail.detail_effect}]及涉及关键人物[{current_detail.key_role}]，
     关键情节[{current_detail.key_plot}]
     
     已经创作的故事:[{stories[-1]}]
     根据用户反馈意见,修改优化本段故事,最大程度满足修改意见,
     只修改内容,输出格式跟原来的故事结构保持一致
     用户反馈意见:[{_detail_modify_feedback}]
     修改后的故事:
     """

    print_msg = "\n\n"
    # print(print_msg)
    # await web_socket.send_text(print_msg)
    stream_response = llm.stream(storyline_detail_modify_prompt)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content
    stories[-1] = response
    return {
        "stories": [stories],
    }


def storyline_detail_human_feedback(state):
    msg = input("内容是否满意?如果满意请直接回复Y,如果不满意请发送修改意见\n")
    if msg.upper().startswith("Y"):
        return {
            "storyline_detail_human_feedback": "",
        }

    return {
        "storyline_detail_human_feedback": msg,
    }


def storyline_detail_finish_node(state):
    # web_socket = state.get("web_socket")
    # await web_socket.send_text(CHAT_STREAM_SERVE_DONE)
    print("\n--storyline_detail_finish_node--\n")
    return {"storyline_detail_human_flag": state.get("storyline_detail_human_flag")}


NODE_GENERATE_DETAIL = "generate_storyline_details"
NODE_MODIFY_DETAIL = "modify_storyline_detail"
NODE_DETAIL_HUMAN_FEEDBACK = "storyline_detail_human_router"
NODE_DETAIL_FINISH = "storyline_detail_finish_node"


def detail_router(state):
    storyline_detail_human_flag = state.get("storyline_detail_human_flag")
    if storyline_detail_human_flag:
        return NODE_STORYLINE_HUMAN_FEEDBACK

    return NODE_DETAIL_FINISH


def storyline_detail_human_router(state):
    _detail_feedback = state.get("storyline_detail_human_feedback")
    if _detail_feedback:
        return NODE_MODIFY_DETAIL
    return NODE_DETAIL_FINISH


def get_detail_graph():
    workflow = StateGraph(StorylineDetailState)
    workflow.add_node(NODE_GENERATE_DETAIL, generate_storyline_details)
    workflow.add_node(NODE_MODIFY_DETAIL, modify_storyline_detail)
    workflow.add_node(NODE_DETAIL_FINISH, storyline_detail_finish_node)

    workflow.set_entry_point(NODE_GENERATE_DETAIL)
    workflow.set_finish_point(NODE_DETAIL_FINISH)

    workflow.add_conditional_edges(
        NODE_GENERATE_DETAIL,
        detail_router,
        {
            NODE_STORYLINE_HUMAN_FEEDBACK: NODE_STORYLINE_HUMAN_FEEDBACK,
        },
    )


def story_router(state):
    storyline_human_flag = state.get("storyline_human_flag")
    if storyline_human_flag:
        return NODE_STORYLINE_HUMAN_FEEDBACK

    return NODE_STORYLINE_GENERATE_FINISH


def storyline_human_router(state):
    storyline_human_feedback = state.get("storyline_human_feedback")
    if storyline_human_feedback:
        return NODE_MODIFY_STORYLINE
    return NODE_STORYLINE_GENERATE_FINISH


def get_storyline_graph():
    workflow = StateGraph(StorylineState)
    workflow.add_node(NODE_GENERATE_STORYLINE, generate_storyline_node)
    workflow.add_node(NODE_MODIFY_STORYLINE, modify_storyline_node)
    workflow.add_node(NODE_STORYLINE_HUMAN_FEEDBACK, storyline_human_feedback_node)
    workflow.add_node(NODE_STORYLINE_GENERATE_FINISH, generate_storyline_finish_node)

    workflow.set_entry_point(NODE_GENERATE_STORYLINE)
    workflow.set_finish_point(NODE_STORYLINE_GENERATE_FINISH)

    workflow.add_conditional_edges(
        NODE_GENERATE_STORYLINE,
        story_router,
        {
            NODE_STORYLINE_HUMAN_FEEDBACK: NODE_STORYLINE_HUMAN_FEEDBACK,
            NODE_STORYLINE_GENERATE_FINISH: NODE_STORYLINE_GENERATE_FINISH
        }
    )
    workflow.add_conditional_edges(
        NODE_STORYLINE_HUMAN_FEEDBACK,
        storyline_human_router,
        {
            NODE_MODIFY_STORYLINE: NODE_MODIFY_STORYLINE,
            NODE_STORYLINE_GENERATE_FINISH: NODE_STORYLINE_GENERATE_FINISH
        }
    )
    workflow.add_edge(NODE_MODIFY_STORYLINE, NODE_STORYLINE_HUMAN_FEEDBACK)

    return workflow


async def test_storyline():
    app = get_storyline_graph().compile()
    print(app.get_graph().draw_mermaid())


async def test_demo():
    app = get_background_graph().compile()

    # print(app.get_graph().draw_mermaid())
    llm = LLMFactory.get_llm(LLMType.DEEPSEEK)
    inputs = {
        "llm": llm,
        "idea": "10年后的世界",
        "background_human_flag": True
    }
    await app.ainvoke(input=inputs)
    # for event in events:
    #     print("--------")
    #     print(event)


async def test_global():
    workflow = StateGraph(StoryState)
    background_graph = get_background_graph()
    storyline_graph = get_storyline_graph()

    workflow.add_node("background_graph", background_graph.compile())
    workflow.add_node("storyline_graph", storyline_graph.compile())

    workflow.set_entry_point("background_graph")
    workflow.add_edge("background_graph", "storyline_graph")
    workflow.set_finish_point("storyline_graph")

    main_app = workflow.compile()
    # print(main_app.get_graph(xray=1).draw_mermaid())

    llm = LLMFactory.get_llm(LLMType.DEEPSEEK)
    inputs = {
        "llm": llm,
        "idea": "10年后的世界",
        "background_human_flag": True,
        "storyline_human_flag": True
    }
    await main_app.ainvoke(input=inputs)


if __name__ == "__main__":
    # asyncio.run(test_demo())
    # asyncio.run(test_storyline())
    asyncio.run(test_global())
    # pydantic_parse = PydanticOutputParser(pydantic_object=StoryLine)
    # model_format_instructions = pydantic_parse.get_format_instructions(),
    # print(model_format_instructions)
