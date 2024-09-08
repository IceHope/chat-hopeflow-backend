import asyncio
import json
import os
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import APIRouter
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket

from models.factory.llm_factory import LLMFactory
from models.model_type import LLMType
from schema.agent_schema import StoryLineAgentSchema
from utils.command_constants import CHAT_STREAM_SERVE_DONE

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "storyLine" + datetime.now().strftime(
    "%d/%m/%Y %H:%M:%S"
)

storyline_router = APIRouter()


class Country(BaseModel):
    country_name: str = Field(description="国家或者地区的名称", alias="名称")
    country_introduction: str = Field(
        description="环境描写，科技水平，信仰情况等关键信息介绍", alias="关键信息"
    )


class Background(BaseModel):
    world_name: str = Field(description="世界的名称", alias="世界名称")
    main_countries: list[Country] = Field(
        description="世界的主要国家或地区分布列表", alias="主要国家或地区"
    )
    background_story: list[str] = Field(
        description="世界背景故事,需要以时间线的形式描述世界的主要历史沿革，国家或地区之间的重大事件及带来的影响变化等",
        alias="世界背景故事",
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
    storyline: str
    storyline_details: list[StoryLineDetail]
    background_human_flag: bool
    storyline_human_flag: bool
    detail_human_flag: bool


class BackgroundState(StoryState):
    background_human_feedback: str


class StorylineState(StoryState):
    storyline_human_feedback: str


class StorylineDetailState(StoryState):
    detail_human_feedback: str
    current_detail_index: int
    stories: list[str]


KEY_LLM = "llm"
KEY_WEB_SOCKET = "web_socket"
KEY_IDEA = "idea"
KEY_BACKGROUND = "background"
KEY_STORYLINE = "storyline"
KEY_STORYLINE_DETAILS = "storyline_details"
KEY_BACKGROUND_HUMAN_FLAG = "background_human_flag"
KEY_STORYLINE_HUMAN_FLAG = "storyline_human_flag"
KEY_DETAIL_HUMAN_FLAG = "detail_human_flag"
KEY_BACKGROUND_HUMAN_FEEDBACK = "background_human_feedback"
KEY_STORYLINE_HUMAN_FEEDBACK = "storyline_human_feedback"
KEY_DETAIL_HUMAN_FEEDBACK = "detail_human_feedback"
KEY_CURRENT_DETAIL_INDEX = "current_detail_index"
KEY_STORIES = "stories"


async def generate_background(state):
    llm = state.get(KEY_LLM)
    web_socket = state.get(KEY_WEB_SOCKET)
    idea = state.get(KEY_IDEA)

    pydantic_parse = PydanticOutputParser(pydantic_object=Background)
    model_format_instructions = (pydantic_parse.get_format_instructions(),)

    background_prompt = f"""请根据故事灵感[{idea}]创作故事的世界信息和背景故事，其中：
    世界信息需要包括世界的主要国家或地区分布，不同国家或地区的环境描写，科技水平，信仰情况等
    世界背景故事需要以时间线的形式描述世界的主要历史沿革，国家或地区之间的重大事件及带来的影响变化等.
    {model_format_instructions}
    """

    stream_response = llm.stream(background_prompt)

    print_msg = "# " + idea + "\n" + "## 世界观背景故事\n"
    print(print_msg)
    await web_socket.send_text(print_msg)

    response = ""
    for chunk in stream_response:
        # print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))

        response += chunk.content

    return {
        KEY_BACKGROUND: response,
    }


async def check_background_human_feedback(state):
    web_socket = state.get(KEY_WEB_SOCKET)
    # 等待前端的反馈
    await web_socket.send_text(
        "\n #### 生成的世界观背景故事是否满意，如果满意请直接回复Y,如果不满意请发送修改意见"
    )
    await web_socket.send_text(CHAT_STREAM_SERVE_DONE)

    chat_data = await web_socket.receive_text()

    feedback = StoryLineAgentSchema(**json.loads(chat_data)).data

    print("\nHuman feedback:", feedback)
    return {
        KEY_BACKGROUND_HUMAN_FEEDBACK: (
            "" if feedback.upper().startswith("Y") else feedback
        )
    }


async def modify_background(state):
    llm = state.get(KEY_LLM)
    web_socket = state.get(KEY_WEB_SOCKET)
    idea = state.get(KEY_IDEA)
    background = state.get(KEY_BACKGROUND)
    _feedback = state.get(KEY_BACKGROUND_HUMAN_FEEDBACK)

    modify_prompt = f"""已知故事灵感[{idea}],请根据用户得反馈意见,对已经创作的背景故事进行修改,
            只修改用户反馈的内容,不要修改其他无关的,返回格式跟原来保持一致,
            已经创作的背景信息:[{background}],
            用户修改意见:[{_feedback}],
            修改后的背景信息:
            """

    stream_response = llm.stream(modify_prompt)

    print_msg = "# " + idea + "\n" + "## 世界观背景故事(修改后)\n"
    print(print_msg)
    await web_socket.send_text(print_msg)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))

        response += chunk.content

    return {
        KEY_BACKGROUND: response,
    }


NODE_GENERATE_BACKGROUND = "generate_background"
NODE_MODIFY_BACKGROUND = "modify_background"
NODE_BACKGROUND_HUMAN_FEEDBACK = "check_background_human_feedback"


def background_router(state):
    background_human_flag = state.get(KEY_BACKGROUND_HUMAN_FLAG)
    if background_human_flag:
        return NODE_BACKGROUND_HUMAN_FEEDBACK

    return END


def background_human_router(state):
    background_human_feedback = state.get(KEY_BACKGROUND_HUMAN_FEEDBACK)
    if background_human_feedback:
        return NODE_MODIFY_BACKGROUND
    return END


def get_background_graph():
    workflow = StateGraph(BackgroundState)

    workflow.add_node(NODE_GENERATE_BACKGROUND, generate_background)
    workflow.add_node(NODE_MODIFY_BACKGROUND, modify_background)
    workflow.add_node(NODE_BACKGROUND_HUMAN_FEEDBACK, check_background_human_feedback)

    workflow.set_entry_point(NODE_GENERATE_BACKGROUND)

    workflow.add_conditional_edges(
        NODE_GENERATE_BACKGROUND,
        background_router,
        {NODE_BACKGROUND_HUMAN_FEEDBACK: NODE_BACKGROUND_HUMAN_FEEDBACK, END: END},
    )
    workflow.add_conditional_edges(
        NODE_BACKGROUND_HUMAN_FEEDBACK,
        background_human_router,
        {NODE_MODIFY_BACKGROUND: NODE_MODIFY_BACKGROUND, END: END},
    )
    workflow.add_edge(NODE_MODIFY_BACKGROUND, NODE_BACKGROUND_HUMAN_FEEDBACK)
    return workflow


async def generate_storyline(state: StoryState):
    llm = state.get(KEY_LLM)
    web_socket = state.get(KEY_WEB_SOCKET)
    idea = state.get(KEY_IDEA)
    background = state.get(KEY_BACKGROUND)

    storyline_prompt = """请根据世界观背景故事[{background}]，围绕故事灵感[{idea}]，创作故事的关键情节线安排,
           用中文回答,只输出回复的内容,不要输出无关内容,包括解释说明
           {model_format_instructions}
           """
    pydantic_parse = PydanticOutputParser(pydantic_object=StoryLine)
    prompt = storyline_prompt.format(
        background=background,
        idea=idea,
        model_format_instructions=pydantic_parse.get_format_instructions(),
    )

    print_msg = "\n## 故事情节大纲\n"
    # print(print_msg)
    await web_socket.send_text(print_msg)

    stream_response = llm.stream(prompt)
    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content

    # fix_parse = OutputFixingParser.from_llm(parser=pydantic_parse, llm=deep_seek_llm)
    storyline = pydantic_parse.parse(response)
    # print(type(storyline))
    # pprint.pprint(storyline.model_dump_json(by_alias=True))

    return {
        KEY_STORYLINE: storyline.model_dump_json(by_alias=True),
        KEY_STORYLINE_DETAILS: storyline.detailed_story_line,
    }


async def modify_storyline(state):
    llm = state.get(KEY_LLM)
    web_socket = state.get(KEY_WEB_SOCKET)
    idea = state.get(KEY_IDEA)
    background = state.get(KEY_BACKGROUND)
    storyline = state.get(KEY_STORYLINE)
    _story_feedback = state.get(KEY_STORYLINE_HUMAN_FEEDBACK)
    pydantic_parse = PydanticOutputParser(pydantic_object=StoryLine)
    model_format_instructions = pydantic_parse.get_format_instructions()

    prompt = f"""
       目前已知的信息有:
       故事灵感: {idea}
       世界观背景故事:{background}
       创作故事的关键情节线安排:{storyline}
       针对情节安排,用户提出了修改意见,请根据反馈意见,结合已知的信息,对情节安排进行优化修改
       用中文回答,只输出回复的内容,不要输出无关内容,包括解释说明
       用户没有反馈的地方,不要修改,格式跟原来的保持一致:
        {model_format_instructions}
       用户的反馈意见:[{_story_feedback}]
      
       你最终的修改:
       """
    stream_response = llm.stream(prompt)

    print_msg = "## 故事情节大纲(修改后)\n"
    print(print_msg)
    await web_socket.send_text(print_msg)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content

    storyline = pydantic_parse.parse(response)
    return {
        KEY_STORYLINE: storyline,
    }


async def check_storyline_human_feedback(state):
    web_socket = state.get(KEY_WEB_SOCKET)
    # 等待前端的反馈
    await web_socket.send_text(
        "\n #### 生成的内容是否满意，如果满意请直接回复Y,如果不满意请发送修改意见"
    )
    await web_socket.send_text(CHAT_STREAM_SERVE_DONE)

    chat_data = await web_socket.receive_text()
    print("\nHuman feedback:", chat_data)

    _feedback = StoryLineAgentSchema(**json.loads(chat_data)).data

    return {
        KEY_STORYLINE_HUMAN_FEEDBACK: (
            "" if _feedback.upper().startswith("Y") else _feedback
        ),
    }


NODE_GENERATE_STORYLINE = "generate_storyline"
NODE_MODIFY_STORYLINE = "modify_storyline"
NODE_STORYLINE_HUMAN_FEEDBACK = "check_storyline_human_feedback"


def story_router(state):
    storyline_human_flag = state.get(KEY_STORYLINE_HUMAN_FLAG)
    if storyline_human_flag:
        return NODE_STORYLINE_HUMAN_FEEDBACK

    return END


def storyline_human_router(state):
    storyline_human_feedback = state.get(KEY_STORYLINE_HUMAN_FEEDBACK)
    if storyline_human_feedback:
        return NODE_MODIFY_STORYLINE
    return END


def get_storyline_graph():
    workflow = StateGraph(StorylineState)
    workflow.add_node(NODE_GENERATE_STORYLINE, generate_storyline)
    workflow.add_node(NODE_MODIFY_STORYLINE, modify_storyline)
    workflow.add_node(NODE_STORYLINE_HUMAN_FEEDBACK, check_storyline_human_feedback)

    workflow.set_entry_point(NODE_GENERATE_STORYLINE)

    workflow.add_conditional_edges(
        NODE_GENERATE_STORYLINE,
        story_router,
        {NODE_STORYLINE_HUMAN_FEEDBACK: NODE_STORYLINE_HUMAN_FEEDBACK, END: END},
    )
    workflow.add_conditional_edges(
        NODE_STORYLINE_HUMAN_FEEDBACK,
        storyline_human_router,
        {NODE_MODIFY_STORYLINE: NODE_MODIFY_STORYLINE, END: END},
    )
    workflow.add_edge(NODE_MODIFY_STORYLINE, NODE_STORYLINE_HUMAN_FEEDBACK)

    return workflow


async def generate_detail(state):
    llm = state.get(KEY_LLM)
    web_socket = state.get(KEY_WEB_SOCKET)
    idea = state.get(KEY_IDEA)
    background = state.get(KEY_BACKGROUND)
    storyline = state.get(KEY_STORYLINE)
    storyline_details = state.get(KEY_STORYLINE_DETAILS)

    current_detail_index = state.get(KEY_CURRENT_DETAIL_INDEX)
    current_detail_index = 0 if current_detail_index is None else current_detail_index
    print(current_detail_index)

    stories = state.get(KEY_STORIES)
    stories = [] if stories is None else stories

    last_block_content = ""
    if stories is not None and len(stories) > 0:
        # 在这里取上一段落的最后50个字，可根据需要修改保留的长度
        keep_length = 100
        last_block = stories[-1][(-1 * keep_length) :]
        last_block_content = (
            f"创作时需要承接[上一段落的末尾:{last_block}，确保表达的连贯性"
        )

    current_detail = storyline_details[current_detail_index]

    storyline_detail_prompt = f"""请根据故事灵感[{idea}],世界观背景故事[{background}]，故事情节线安排大纲[{storyline}]，
                 继续创作具体的故事情节内容,{last_block_content}要求如下:
                 根据本段故事的作用[{current_detail.detail_effect}]及涉及关键人物[{current_detail.key_role}]，
                 将关键情节[{current_detail.key_plot}]扩写为完整的故事,
                 每段故事需要尽量包括行动描写、心理活动描写和对白等细节,
                 每次创作只是完整文章结构中的一部分，承担本段故事作用说明的作用任务，只需要按要求完成关键情节的描述即可，不需要考虑本段故事自身结构的完整性.
                 用中文回答,只输出回复的内容,不要输出无关内容,包括解释说明
                 输出的格式限定如下:
                 ### {current_detail.detail_effect}\n
                 [具体内容]
                 """

    print_msg = "\n\n"
    # print(print_msg)
    await web_socket.send_text(print_msg)
    stream_response = llm.stream(storyline_detail_prompt)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content

    response = "\n\n" + response
    stories.append(response)
    return {KEY_STORIES: stories, KEY_CURRENT_DETAIL_INDEX: current_detail_index + 1}


async def modify_detail(state):
    llm = state.get(KEY_LLM)
    web_socket = state.get(KEY_WEB_SOCKET)
    idea = state.get(KEY_IDEA)
    background = state.get(KEY_BACKGROUND)
    storyline = state.get(KEY_STORYLINE)
    storyline_details = state.get(KEY_STORYLINE_DETAILS)
    current_detail_index = state.get(KEY_CURRENT_DETAIL_INDEX)
    stories = state.get(KEY_STORIES)

    current_detail = storyline_details[current_detail_index - 1]

    _detail_modify_feedback = state.get(KEY_DETAIL_HUMAN_FEEDBACK)

    storyline_detail_modify_prompt = f"""
     已知的信息:故事灵感[{idea}],世界观背景故事[{background}]，故事情节线安排大纲[{storyline}]，

     本段故事的作用[{current_detail.detail_effect}]及涉及关键人物[{current_detail.key_role}]，
     关键情节[{current_detail.key_plot}]

     已经创作的故事:[{stories[-1]}]
     根据用户反馈意见,修改优化本段故事,最大程度满足修改意见,
     只修改内容,输出格式跟原来的故事结构保持一致
     用户反馈意见:[{_detail_modify_feedback}]
     修改后的故事:
     """

    print_msg = "\n\n"
    print(print_msg)
    await web_socket.send_text(print_msg)

    stream_response = llm.stream(storyline_detail_modify_prompt)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content

    stories[-1] = "\n\n" + response
    return {KEY_STORIES: stories}


async def check_detail_human_feedback(state):
    web_socket = state.get(KEY_WEB_SOCKET)
    # 等待前端的反馈
    await web_socket.send_text(
        "\n #### 当前情节是否满意，如果满意请直接回复Y,如果不满意请发送修改意见"
    )
    await web_socket.send_text(CHAT_STREAM_SERVE_DONE)

    chat_data = await web_socket.receive_text()
    print("\nHuman feedback:", chat_data)

    feedback = StoryLineAgentSchema(**json.loads(chat_data)).data
    return {
        KEY_DETAIL_HUMAN_FEEDBACK: (
            "" if feedback.upper().startswith("Y") else feedback
        )
    }


async def generate_finish(state):
    web_socket = state.get(KEY_WEB_SOCKET)
    idea = state.get(KEY_IDEA)
    stories = state.get(KEY_STORIES)

    await web_socket.send_text(CHAT_STREAM_SERVE_DONE)

    await web_socket.send_text(f"创作完成,下面是全文\n\n")
    await web_socket.send_text(CHAT_STREAM_SERVE_DONE)

    await web_socket.send_text(f"# {idea}\n\n")
    await web_socket.send_text(stories)
    await web_socket.send_text(CHAT_STREAM_SERVE_DONE)

    print("\n--generate_finish--\n")
    return {KEY_CURRENT_DETAIL_INDEX: 0}


NODE_GENERATE_DETAIL = "generate_detail"
NODE_MODIFY_DETAIL = "modify_detail"
NODE_DETAIL_HUMAN_FEEDBACK = "check_detail_human_feedback"
NODE_DETAIL_FINISH = "generate_finish"


def detail_router(state):
    detail_human_flag = state.get(KEY_DETAIL_HUMAN_FLAG)
    if detail_human_flag:
        return NODE_DETAIL_HUMAN_FEEDBACK

    if state.get(KEY_CURRENT_DETAIL_INDEX) == len(state.get(KEY_STORYLINE_DETAILS)):
        return NODE_DETAIL_FINISH

    return NODE_GENERATE_DETAIL


def detail_human_router(state):
    _detail_feedback = state.get(KEY_DETAIL_HUMAN_FEEDBACK)

    if _detail_feedback:
        return NODE_MODIFY_DETAIL

    if state.get(KEY_CURRENT_DETAIL_INDEX) == len(state.get(KEY_STORYLINE_DETAILS)):
        return NODE_DETAIL_FINISH

    return NODE_GENERATE_DETAIL


def get_detail_graph():
    workflow = StateGraph(StorylineDetailState)
    workflow.add_node(NODE_GENERATE_DETAIL, generate_detail)
    workflow.add_node(NODE_DETAIL_HUMAN_FEEDBACK, check_detail_human_feedback)
    workflow.add_node(NODE_MODIFY_DETAIL, modify_detail)
    workflow.add_node(NODE_DETAIL_FINISH, generate_finish)

    workflow.set_entry_point(NODE_GENERATE_DETAIL)
    workflow.set_finish_point(NODE_DETAIL_FINISH)

    workflow.add_conditional_edges(
        NODE_GENERATE_DETAIL,
        detail_router,
        {
            NODE_DETAIL_HUMAN_FEEDBACK: NODE_DETAIL_HUMAN_FEEDBACK,
            NODE_DETAIL_FINISH: NODE_DETAIL_FINISH,
            NODE_GENERATE_DETAIL: NODE_GENERATE_DETAIL,
        },
    )
    workflow.add_conditional_edges(
        NODE_DETAIL_HUMAN_FEEDBACK,
        detail_human_router,
        {
            NODE_MODIFY_DETAIL: NODE_MODIFY_DETAIL,
            NODE_DETAIL_FINISH: NODE_DETAIL_FINISH,
            NODE_GENERATE_DETAIL: NODE_GENERATE_DETAIL,
        },
    )
    workflow.add_edge(NODE_MODIFY_DETAIL, NODE_DETAIL_HUMAN_FEEDBACK)
    return workflow


def get_parent_graph():
    workflow = StateGraph(StoryState)
    background_graph = get_background_graph()
    storyline_graph = get_storyline_graph()
    detail_graph = get_detail_graph()

    workflow.add_node("generate_background_graph", background_graph.compile())
    workflow.add_node("generate_storyline_graph", storyline_graph.compile())
    workflow.add_node("generate_detail_graph", detail_graph.compile())

    workflow.set_entry_point("generate_background_graph")
    workflow.add_edge("generate_background_graph", "generate_storyline_graph")
    workflow.add_edge("generate_storyline_graph", "generate_detail_graph")
    workflow.set_finish_point("generate_detail_graph")

    return workflow
    # # print(main_app.get_graph(xray=1).draw_mermaid())
    #
    # llm = LLMFactory.get_llm(LLMType.DEEPSEEK)
    # inputs = {
    #     "llm": llm,
    #     "idea": "10年后的世界",
    #     "background_human_flag": True,
    #     "storyline_human_flag": True
    # }
    # await main_app.ainvoke(input=inputs)


@storyline_router.websocket("/ws/agent/storyline")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        print("connection open")
        while True:
            try:
                request_msg = await websocket.receive_text()
                print(f"Received message: {request_msg}")

                chat_request_data = StoryLineAgentSchema(**json.loads(request_msg))
                llm = LLMFactory.get_llm(
                    mode_type=LLMType.get_enum_from_value(chat_request_data.model_type),
                    mode_name=chat_request_data.model_name,
                )
                inputs = {
                    "llm": llm,
                    "web_socket": websocket,
                    "idea": chat_request_data.data,
                    "background_human_flag": chat_request_data.background_human_flag,
                    "storyline_human_flag": chat_request_data.storyline_human_flag,
                    "detail_human_flag": chat_request_data.detail_human_flag,
                }

                print("inputs: ", inputs)
                app = get_parent_graph().compile()
                await app.ainvoke(input=inputs)

            except Exception as e:
                print(f"Error while receiving or processing data: {e}")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing connection")
        await websocket.close()


async def test_demo():
    # test_parse()
    app = get_parent_graph().compile()

    print(app.get_graph().draw_mermaid())

    # events = app.stream({"idea": "100年后的世界"})
    # for event in events:
    #     print("--------")
    #     print(event)


if __name__ == "__main__":
    # asyncio.run(test_demo())
    # background_graph = get_background_graph().compile()
    # storyline_graph = get_storyline_graph().compile()
    # detail_graph = get_detail_graph().compile()
    # print(background_graph.get_graph().draw_mermaid())
    # print(storyline_graph.get_graph().draw_mermaid())
    # print(detail_graph.get_graph().draw_mermaid())

    parent_graph = get_parent_graph().compile()
    print(parent_graph.get_graph().draw_mermaid())
