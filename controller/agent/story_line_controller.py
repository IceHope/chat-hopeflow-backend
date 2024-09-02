import asyncio
import json
import operator
import os
from datetime import datetime
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from fastapi import APIRouter
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket

from factory.llm_factory import LLMFactory
from factory.mode_type import LLMType
from schema.agent_schema import StoryLineAgentSchema
from utils.constants import COMMAND_DONE_FROM_SERVE

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "StoryLine" + datetime.now().strftime(
    "%d/%m/%Y %H:%M:%S"
)

storyline_router = APIRouter()

openai_llm = LLMFactory.get_llm(LLMType.OPENAI, "gpt-3.5-turbo")
deep_seek_llm = LLMFactory.get_llm(LLMType.DEEPSEEK)


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
    web_socket: WebSocket
    idea: str
    background: str
    storyline: str
    storyline_details: list[StoryLineDetail]
    current_detail_index: int
    stories: Annotated[list[str], operator.add]


async def generate_background(state: StoryState):
    web_socket = state.get("web_socket")
    idea = state.get("idea")
    background_prompt = """请根据故事灵感[{idea}],创作故事的世界信息和背景故事,中文回答
    {model_format_instructions}
    """
    pydantic_parse = PydanticOutputParser(pydantic_object=Background)
    prompt = background_prompt.format(
        idea=idea, model_format_instructions=pydantic_parse.get_format_instructions()
    )

    print_msg = "# " + idea + "\n" + "## 世界观背景故事\n"
    print(print_msg)
    await web_socket.send_text(print_msg)

    stream_response = openai_llm.stream(prompt)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))

        response += chunk.content

    # fix_parse = OutputFixingParser.from_llm(parser=pydantic_parse, llm=deep_seek_llm)
    background = pydantic_parse.parse(response).model_dump_json()

    return {"background": background}


async def generate_storyline(state: StoryState):
    web_socket = state.get("web_socket")
    idea = state.get("idea")
    background = state.get("background")

    storyline_prompt = """请根据世界观背景故事[{background}]，围绕故事灵感[{idea}]，创作故事的关键情节线安排,中文回答
           {model_format_instructions}
           """
    pydantic_parse = PydanticOutputParser(pydantic_object=StoryLine)
    prompt = storyline_prompt.format(
        background=background,
        idea=idea,
        model_format_instructions=pydantic_parse.get_format_instructions(),
    )

    print_msg = "\n## 关键情节线安排\n"
    print(print_msg)
    await web_socket.send_text(print_msg)

    stream_response = openai_llm.stream(prompt)
    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content

    # fix_parse = OutputFixingParser.from_llm(parser=pydantic_parse, llm=deep_seek_llm)
    storyline = pydantic_parse.parse(response)
    return {
        "storyline": storyline.model_dump_json(),
        "storyline_details": storyline.detailed_story_line,
        "current_detail_index": 0,
        "stories": [],
    }


async def generate_storyline_detail(state: StoryState):
    web_socket = state.get("web_socket")
    idea = state.get("idea")
    background = state.get("background")
    storyline = state.get("storyline")

    storyline_details = state.get("storyline_details")
    current_detail_index = state.get("current_detail_index")
    stories = state.get("stories")

    current_detail = storyline_details[current_detail_index]

    storyline_detail_prompt = f"""请根据故事灵感[{idea}],世界观背景故事[{background}]，故事情节线安排[{storyline}]，
           参考之前已经创作的内容[{stories}],继续创作具体的故事情节内容,要求如下:
           根据本段故事的作用[{current_detail.detail_effect}]及涉及关键人物[{current_detail.key_role}]，将关键情节[{current_detail.key_plot}]扩写为完整的故事,
           每段故事需要尽量包括行动描写、心理活动描写和对白等细节,
           每次创作只是完整文章结构中的一部分，承担本段故事作用说明的作用任务，只需要按要求完成关键情节的描述即可，不需要考虑本段故事自身结构的完整性.
           输出的格式限定如下:
           ### {current_detail.detail_effect}\n
           [具体内容]
           """

    print_msg = "\n\n"
    print(print_msg)
    await web_socket.send_text(print_msg)
    stream_response = openai_llm.stream(storyline_detail_prompt)

    response = ""
    for chunk in stream_response:
        print(chunk.content, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk.content))
        response += chunk.content

    return {
        "stories": [response],
        "current_detail_index": current_detail_index + 1,
    }


async def generate_finish(state: StoryState):
    web_socket = state.get("web_socket")
    await web_socket.send_text(COMMAND_DONE_FROM_SERVE)
    print("\n--generate_finish--\n")
    return {"current_detail_index": state.get("current_detail_index") - 1}


def generate_router(state: StoryState):
    if state.get("current_detail_index") == len(state.get("storyline_details")):
        return "generate_finish"
    else:
        return "generate_storyline_detail"


def get_graph():
    workflow = StateGraph(StoryState)
    workflow.add_node("generate_background", generate_background)
    workflow.add_node("generate_storyline", generate_storyline)
    workflow.add_node("generate_storyline_detail", generate_storyline_detail)
    workflow.add_node("generate_finish", generate_finish)

    workflow.set_entry_point("generate_background")
    workflow.set_finish_point("generate_finish")

    workflow.add_edge("generate_background", "generate_storyline")
    workflow.add_edge("generate_storyline", "generate_storyline_detail")
    workflow.add_conditional_edges(
        "generate_storyline_detail",
        generate_router,
        {
            "generate_finish": "generate_finish",
            "generate_storyline_detail": "generate_storyline_detail",
        },
    )
    return workflow


app = get_graph().compile()


@storyline_router.websocket("/ws/agent/storyline")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        print("connection open")
        while True:
            try:
                message = await websocket.receive_text()
                print(f"Received message: {message}")

                message_data = json.loads(message)
                msg = StoryLineAgentSchema(**message_data)
                inputs = {"web_socket": websocket, "idea": msg.data}

                print("inputs: ", inputs)
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
    app = get_graph().compile()

    # print(app.get_graph().draw_mermaid())

    events = app.stream({"idea": "100年后的世界"})
    for event in events:
        print("--------")
        print(event)


if __name__ == "__main__":
    asyncio.run(test_demo())
