import asyncio
import json
import time
import traceback
from typing import List

from fastapi import APIRouter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse.decorators import observe, langfuse_context
from pydantic import BaseModel
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from controller.question_prompt import QUESTION_PROMPT
from dao.redis_dao import ChatRedisManager
from factory.llm_factory import LLMFactory
from factory.mode_type import LLMType
from schema.chat_schema import ChatRequestData
from schema.custom_model import get_all_custom_models
from utils.constants import COMMAND_DONE_FROM_SERVE, COMMAND_STOP_FROM_CLIENT
from utils.log_utils import LogUtils

chat_router = APIRouter()

chat_manager = ChatRedisManager()

follow_question_llm = LLMFactory.get_llm(LLMType.ZHIPU, "GLM-4-Flash")


def generate_follow_questions(ask, reply):
    prompt = ChatPromptTemplate.from_template(QUESTION_PROMPT)
    # 这里可以选择特定的大模型
    chain = prompt | follow_question_llm | StrOutputParser()
    follow_questions = chain.invoke({"number": 3, "ask": ask, "ai_answer": reply})
    LogUtils.log_info(f"follow_questions: {follow_questions}")
    return follow_questions


def generate_image_content(data: str, image_urls: List[str]) -> list:
    """Generate image content.
    这种格式的,即便作为聊天记录的上下文,大部分不支持多模态的国产模型,也会出错,智谱,千问,比较好
    可以看到模型能力的bug
    """
    data_list = [{"type": "text", "text": data}]
    images_list = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    data_list.extend(images_list)
    return data_list


@observe()
async def send_streaming_data(chat_request_data, websocket, response, llm):
    final_result = ""
    start_time = time.time()
    LogUtils.log_info(f"start_time: {start_time}")

    try:
        for chunk in response:
            delta = chunk.content
            if delta is not None:
                if final_result == "":
                    LogUtils.log_info(f"start_stream_time: {time.time() - start_time}")

                final_result += str(delta)
                await websocket.send_text(str(delta))
            await asyncio.sleep(0)  # 让出控制权

    finally:
        outputs = [
            {
                "role": "assistant",
                "content": final_result,
                "model_name": chat_request_data.model_name,
            }
        ]
        chat_manager.add_chat_record(chat_request_data.user_name, chat_request_data.session_id, outputs)
        follow_questions = generate_follow_questions(chat_request_data.data, final_result)
        LogUtils.log_info(f"end_stream_time: {time.time() - start_time}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(COMMAND_DONE_FROM_SERVE)

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(follow_questions)


def get_chat_history(user_name, session_id):
    history = chat_manager.get_history_record(user_name, session_id)
    LogUtils.log_info("history:")
    LogUtils.log_info(history)
    return history


@observe()
async def perform_chat(websocket, chat_request_data: ChatRequestData):
    langfuse_context.update_current_trace(
        name=chat_request_data.user_name,
        user_id=chat_request_data.session_id,
    )
    langfuse_handler = langfuse_context.get_current_langchain_handler()

    llm = LLMFactory.get_llm(
        mode_type=LLMType.get_enum_from_value(chat_request_data.model_type),
        mode_name=chat_request_data.model_name)

    inputs = [{"role": "user", "content": chat_request_data.data}]
    if chat_request_data.image_urls:
        LogUtils.log_info("image_urls: ", chat_request_data.image_urls)

        images_content = generate_image_content(data=chat_request_data.data,
                                                image_urls=chat_request_data.image_urls)
        inputs = [
            {"role": "user",
             "content": images_content},
        ]

    chat_manager.add_chat_record(chat_request_data.user_name, chat_request_data.session_id, inputs)

    if chat_request_data.multi_turn_chat_enabled:
        history = get_chat_history(chat_request_data.user_name, chat_request_data.session_id)
    else:
        history = inputs

    # response = llm.stream(history)
    response = llm.stream(history, config={"callbacks": [langfuse_handler]})
    LogUtils.log_info("response :", response)
    send_task = asyncio.create_task(send_streaming_data(chat_request_data, websocket, response, llm))

    while not send_task.done():
        try:
            # LogUtils.log_info("waiting for message")
            request_msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            LogUtils.log_info("receive new text: ", request_msg)
            if COMMAND_STOP_FROM_CLIENT in request_msg:
                LogUtils.log_info("receive stop command")
                send_task.cancel()  # 取消发送任务
                await asyncio.gather(send_task, return_exceptions=True)  # 等待任务完成或被取消
                LogUtils.log_info("send_stop_flag")
                stop_flag = '<span style="color: #5989F7;"><br><br>[[客户端停止接收信息]]<br><br></span>'
                await websocket.send_text(stop_flag)
                break
        except asyncio.TimeoutError:
            # 没有接收到消息，继续循环
            continue
        except WebSocketDisconnect:
            LogUtils.log_info("Client disconnected")
            break


@observe()
@chat_router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        LogUtils.log_info("connection open ", websocket.client)
        while True:
            try:
                request_msg = await websocket.receive_text()
                LogUtils.log_info("request_msg: ", request_msg)

                chat_request_data = ChatRequestData(**json.loads(request_msg))

                await perform_chat(websocket, chat_request_data)

            except WebSocketDisconnect:
                LogUtils.log_info("Client disconnected")
                break
            except Exception as e:
                LogUtils.log_error(f"Error while receiving or processing data: {e}")
                traceback.print_exc()
                await websocket.send_text(f"Error while receiving or processing data: {e}")
                await websocket.send_text(COMMAND_DONE_FROM_SERVE)

    except Exception as e:
        LogUtils.log_error(f"Error: {e}")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError as e:
                LogUtils.log_error(f"Error while closing websocket: {e}")
        LogUtils.log_info("Connection closed")


@chat_router.get("/chat/modes")
async def get_chat_models():
    return {"models": get_all_custom_models()}


class HistorySnapshots(BaseModel):
    user_name: str


@chat_router.post("/chat/history/snapshots")
async def get_history_snapshots(request: HistorySnapshots):
    LogUtils.log_info("get_history_snapshots")
    LogUtils.log_info("user_name: ", request)
    return chat_manager.get_history_snapshots(request.user_name)


class HistoryRecord(BaseModel):
    user_name: str
    session_id: int


@chat_router.post("/chat/history/record")
async def get_history_record(request: HistoryRecord):
    return chat_manager.get_history_record(request.user_name, request.session_id)
