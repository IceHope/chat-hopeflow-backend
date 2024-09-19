import asyncio
import json
import os
from datetime import datetime

from fastapi import APIRouter
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph
from starlette.websockets import WebSocket

from models.factory.llm_factory import LLMFactory
from models.model_type import LLMType
from schema.chat_schema import ChatRequestData

from utils.command_constants import CHAT_STREAM_SERVE_DONE

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "six_hat" + datetime.now().strftime(
    "%d/%m/%Y %H:%M:%S"
)

from typing import TypedDict

six_hat_router = APIRouter()


def get_completion(
    llm: BaseChatModel,
    prompt: str,
    system_message: str = "You are a helpful assistant.",
):
    messages = [
        ("system", system_message),
        ("user", prompt),
    ]
    response = llm.stream(messages)
    for chunk in response:
        yield chunk.content


class State(TypedDict):
    llm: BaseChatModel
    web_socket: WebSocket
    question: str
    white_hat: str
    red_hat: str
    black_hat: str
    yellow_hat: str
    green_hat: str
    blue_hat: str


async def generate_white_hat(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    question = state.get("question")

    system_message = "你是六顶思维帽模式中的白色帽子,代表事实和数据"
    prompt = f"""
请以完全客观和中立的态度,仅陈述与当前问题相关的事实和数据,不要包含任何个人观点或情感
当前问题:{question}
你的回答:
"""
    completion = get_completion(llm, prompt, system_message=system_message)
    print_msg = "\n## 白色帽子: \n"
    print(print_msg)
    await web_socket.send_text(print_msg)
    text = ""
    for chunk in completion:
        print(chunk, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk))
        text += chunk

    return {"white_hat": text}


async def generate_red_hat(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    question = state.get("question")

    system_message = "你是六顶思维帽模式中的红色帽子,代表情感和直觉"
    prompt = f"""
请基于你的直觉和情感,对当前问题做出即时反应。不需要解释原因,只需表达你的感受和直觉判断
当前问题:{question}
你的回答:
"""
    completion = get_completion(llm, prompt, system_message=system_message)
    print_msg = "\n## 红色帽子: \n"
    print(print_msg)
    await web_socket.send_text(print_msg)
    text = ""
    for chunk in completion:
        print(chunk, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk))
        text += chunk

    return {"red_hat": text}


async def generate_black_hat(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    question = state.get("question")
    white_hat = state.get("white_hat")
    red_hat = state.get("red_hat")

    system_message = "你是六顶思维帽模式中的黑色帽子,代表批判和风险分析"
    prompt = f"""
基于当前问题，考虑到白帽提供的事实和红帽表达的情感，请以批评的角度审视当前问题。指出潜在的风险、缺陷和负面影响，重点关注可能出错的地方
当前问题:{question}
事实:{white_hat}
情感:{red_hat}
你的回答:
"""
    completion = get_completion(llm, prompt, system_message=system_message)
    print_msg = "\n## 黑色帽子: \n"
    print(print_msg)
    await web_socket.send_text(print_msg)
    text = ""
    for chunk in completion:
        print(chunk, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk))
        text += chunk

    return {"black_hat": text}


async def generate_yellow_hat(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    question = state.get("question")
    white_hat = state.get("white_hat")
    red_hat = state.get("red_hat")
    black_hat = state.get("black_hat")

    system_message = "你是六顶思维帽模式中的黄色帽子,代表乐观和机会"
    prompt = f"""
在了解了事实、情感反应和潜在风险后，请以积极乐观的态度看待当前问题。重点关注其中的机遇、优势和潜在收益，提出可能的正面结果
当前问题:{question}
事实:{white_hat}
情感:{red_hat}
批判意见:{black_hat}
你的回答:
"""
    completion = get_completion(llm, prompt, system_message=system_message)
    print_msg = "\n## 黄色帽子: \n"
    print(print_msg)
    await web_socket.send_text(print_msg)
    text = ""
    for chunk in completion:
        print(chunk, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk))
        text += chunk

    return {"yellow_hat": text}


async def generate_green_hat(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    question = state.get("question")
    white_hat = state.get("white_hat")
    red_hat = state.get("red_hat")
    black_hat = state.get("black_hat")
    yellow_hat = state.get("yellow_hat")

    system_message = "你是六顶思维帽模式中的绿色帽子,代表创造性和创新"
    prompt = f"""
基于前面的分析,请跳出常规思维框架,为当前问题提供创新性的解决方案或新颖观点。
不受限于现有条件,大胆设想并提出突破性的创意。考虑如何克服已识别的挑战,同时充分利用潜在机会。
请尽情发挥想象力,提出令人耳目一新的构想
当前问题:{question}
事实:{white_hat}
情感:{red_hat}
批判意见:{black_hat}
积极机遇:{yellow_hat}
你的回答:
"""
    completion = get_completion(llm, prompt, system_message=system_message)
    print_msg = "\n## 绿色帽子: \n"
    print(print_msg)
    await web_socket.send_text(print_msg)
    text = ""
    for chunk in completion:
        print(chunk, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk))
        text += chunk

    return {"green_hat": text}


async def generate_blue_hat(state):
    llm = state.get("llm")
    web_socket = state.get("web_socket")
    question = state.get("question")
    white_hat = state.get("white_hat")
    red_hat = state.get("red_hat")
    black_hat = state.get("black_hat")
    yellow_hat = state.get("yellow_hat")
    green_hat = state.get("green_hat")

    system_message = "你是六顶思维帽模式中的蓝色帽子,代表总结"
    prompt = f"""
基于当前问题,请综合白色帽子的客观事实，红色帽子的情感直觉，以及黑色帽子的批判意见，形成一个完整的背景分析。”
“结合黄色帽子的积极面和机会，以及绿色帽子的创造性解决方案，形成一个整体的解决思路
当前问题:{question}
事实:{white_hat}
情感:{red_hat}
批判意见:{black_hat}
积极机遇:{yellow_hat}
创造性:{green_hat}
你的回答:
"""
    completion = get_completion(llm, prompt, system_message=system_message)
    print_msg = "\n## 蓝色帽子: \n"
    print(print_msg)
    await web_socket.send_text(print_msg)
    text = ""
    for chunk in completion:
        print(chunk, end="", flush=True)
        await asyncio.sleep(0)  # 让出控制权，允许事件循环处理其他任务
        await web_socket.send_text(str(chunk))
        text += chunk

    await web_socket.send_text(CHAT_STREAM_SERVE_DONE)
    return {"blue_hat": text}


# 创建一个工作流对象
workflow = StateGraph(State)

workflow.add_node("generate_white_hat", generate_white_hat)
workflow.add_node("generate_red_hat", generate_red_hat)
workflow.add_node("generate_black_hat", generate_black_hat)
workflow.add_node("generate_yellow_hat", generate_yellow_hat)
workflow.add_node("generate_green_hat", generate_green_hat)
workflow.add_node("generate_blue_hat", generate_blue_hat)

workflow.set_entry_point("generate_white_hat")
workflow.set_finish_point("generate_blue_hat")

workflow.add_edge("generate_white_hat", "generate_red_hat")
workflow.add_edge("generate_red_hat", "generate_black_hat")
workflow.add_edge("generate_black_hat", "generate_yellow_hat")
workflow.add_edge("generate_yellow_hat", "generate_green_hat")
workflow.add_edge("generate_green_hat", "generate_blue_hat")


@six_hat_router.websocket("/ws/agent/six_hat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        print("connection open")
        while True:
            try:
                request_msg = await websocket.receive_text()
                print(f"Received message: {request_msg}")

                chat_request_data = ChatRequestData(**json.loads(request_msg))
                llm = LLMFactory.get_llm(
                    mode_type=LLMType.get_enum_from_value(chat_request_data.model_type),
                    mode_name=chat_request_data.model_name,
                )
                inputs = {
                    "llm": llm,
                    "web_socket": websocket,
                    "question": chat_request_data.data,
                }
                print("inputs: ", inputs)
                app = workflow.compile()
                await app.ainvoke(input=inputs)

            except Exception as e:
                print(f"Error while receiving or processing data: {e}")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing connection")
        await websocket.close()


if __name__ == "__main__":
    print(workflow.compile().get_graph().draw_mermaid())
