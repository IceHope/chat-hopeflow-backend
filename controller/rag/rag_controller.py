import asyncio
import json
import time
import traceback

from fastapi import APIRouter
from langfuse.decorators import observe, langfuse_context
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from controller.question_prompt import generate_follow_questions
from controller.rag.frontend_node import (
    FrontendNodesPayload,
    cast_node_frontend,
    cast_nodes_to_frontend,
)
from controller.rag.request_type import (
    UserNameRequest,
    FileIdKnowledgeRequest,
    FileIdChunkRequest,
)
from dao.knowledge_dao import KnowledgeDao
from rag.rag_base_manager import RagBaseManager, check_image_node
from schema.chat_schema import ChatRequestData
from utils.constants import (
    COMMAND_DONE_FROM_SERVE,
    COMMAND_STOP_FROM_CLIENT,
    COMMAND_STREAM_START_FROM_SERVE,
    RAG_EVENT_GENERATE_START,
    RAG_EVENT_IMAGE_QA_DONE,
    RAG_EVENT_IMAGE_QA_START,
    RAG_RETRIEVE_CHUNK_DONE,
    RAG_RETRIEVE_CHUNK_START,
)
from utils.log_utils import LogUtils

rag_router = APIRouter()

knowledge_dao = KnowledgeDao()

rag_base_manager = RagBaseManager()


@rag_router.post("/rag/knowledge/query_all")
def query_all_knowledge(user: UserNameRequest):
    """查询所有知识"""
    LogUtils.log_info("query_all_knowledge :", user.user_name)
    return knowledge_dao.get_all_knowledges_by_user_name(user.user_name)


@rag_router.post("/rag/knowledge/query_all_chunk_by_file_id")
def query_nodes_by_file_id(file: FileIdKnowledgeRequest):
    """根据文件路径查询节点"""
    LogUtils.log_info("query_nodes_by_file_id :", file.file_id)

    from rag.db.milvus.client import MyMilvusClient

    client = MyMilvusClient()

    return client.search_nodes_from_file_id(
        collect_name=rag_base_manager.get_collection_name(), file_id=file.file_id
    )


@rag_router.post("/rag/vector/query_match_chunk")
def query_chunk_by_file_id(file: FileIdChunkRequest):
    LogUtils.log_info(
        f"query_chunk_by_file_id: query={file.query},file_id={file.file_id}"
    )

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="file_id", value=file.file_id)]
    )

    retrieve_nodes = rag_base_manager.retrieve_chunk(query=file.query, filters=filters)

    frontend_nodes = []
    for node in retrieve_nodes:
        frontend_nodes.append(cast_node_frontend(node))
    return frontend_nodes


@observe()
async def send_streaming_data(chat_request_data, websocket, all_nodes):
    start_time = time.time()

    await websocket.send_text(RAG_EVENT_GENERATE_START)
    LogUtils.log_info(RAG_EVENT_GENERATE_START)

    stream_response = rag_base_manager.generate_chat_stream_response(
        chat_request_data.data, all_nodes
    )
    final_result = ""
    await websocket.send_text(COMMAND_STREAM_START_FROM_SERVE)
    LogUtils.log_info(COMMAND_STREAM_START_FROM_SERVE)

    try:
        for chunk in stream_response.response_gen:
            if chunk is not None:
                final_result += str(chunk)
                await websocket.send_text(str(chunk))
            await asyncio.sleep(0)  # 让出控制权

    finally:

        follow_questions = generate_follow_questions(
            chat_request_data.data, final_result
        )
        LogUtils.log_info(f"end_stream_time: {time.time() - start_time}")

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(COMMAND_DONE_FROM_SERVE)

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(follow_questions)


@observe()
async def perform_chat(websocket, chat_request_data: ChatRequestData):
    langfuse_context.update_current_trace(
        name=chat_request_data.user_name,
        user_id=chat_request_data.session_id,
    )

    origin_query = chat_request_data.data

    start_time = time.time()

    await websocket.send_text(RAG_RETRIEVE_CHUNK_START)
    LogUtils.log_info(RAG_RETRIEVE_CHUNK_START)

    # 1.1 chunk_nodes
    chunk_nodes = await rag_base_manager.aretrieve_chunk(query=origin_query)
    # chunk_nodes = rag_base_manager.retrieve_chunk(query=origin_query)

    chunk_done_time = time.time()
    chunk_cost_time = str(round(chunk_done_time - start_time, 2))

    await websocket.send_text(RAG_RETRIEVE_CHUNK_DONE)
    LogUtils.log_info(RAG_RETRIEVE_CHUNK_DONE, f"{chunk_cost_time} s")

    nodes_payload = FrontendNodesPayload(
        chunk_frontend_nodes=cast_nodes_to_frontend(chunk_nodes)
    )
    nodes_payload_json = json.dumps(
        nodes_payload.__dict__, default=lambda o: o.__dict__
    )
    await websocket.send_text(nodes_payload_json)

    # 2. check image
    text_nodes, image_nodes = check_image_node(chunk_nodes)
    if image_nodes:
        await websocket.send_text(RAG_EVENT_IMAGE_QA_START)
        LogUtils.log_info(RAG_EVENT_IMAGE_QA_START)

        # 2.1 image_nodes
        image_nodes = await rag_base_manager.agenerate_image_nodes_response(
            origin_query, image_nodes
        )

        image_cost_time = str(round(time.time() - chunk_done_time, 2))

        await websocket.send_text(RAG_EVENT_IMAGE_QA_DONE)
        LogUtils.log_info(RAG_EVENT_IMAGE_QA_DONE, f"{image_cost_time} s")

    all_nodes = text_nodes + image_nodes

    # 3. chat

    send_task = asyncio.create_task(
        send_streaming_data(chat_request_data, websocket, all_nodes)
    )

    while not send_task.done():
        try:
            # LogUtils.log_info("waiting for message")
            request_msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            LogUtils.log_info("receive new text: ", request_msg)
            if COMMAND_STOP_FROM_CLIENT in request_msg:
                LogUtils.log_info("receive stop command")
                send_task.cancel()  # 取消发送任务
                await asyncio.gather(
                    send_task, return_exceptions=True
                )  # 等待任务完成或被取消
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
@rag_router.websocket("/ws/rag/chat_query")
async def chat_query_websocket(websocket: WebSocket):
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
                await websocket.send_text(
                    f"Error while receiving or processing data: {e}"
                )
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
