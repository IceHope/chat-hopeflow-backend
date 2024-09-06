import asyncio
import json
import time
import traceback

from fastapi import APIRouter
from langfuse.decorators import observe, langfuse_context
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from controller.question_prompt import generate_follow_questions
from controller.rag.request_type import (
    UserNameRequest,
    FileIdKnowledgeRequest,
    FileIdChunkRequest,
)
from dao.knowledge_dao import KnowledgeDao
from dao.redis_dao import ChatRedisManager
from rag.rag_base_manager import RagBaseManager, check_image_node
from schema.chat_schema import ChatRequestData
from schema.frontend_node import (
    FrontendNodesPayload,
    cast_node_frontend,
    cast_nodes_to_frontend,
)
from schema.rag_config import RagFrontendConfig
from utils.command_constants import *
from utils.log_utils import LogUtils

rag_router = APIRouter()

knowledge_dao = KnowledgeDao()

rag_base_manager = RagBaseManager()

chat_manager = ChatRedisManager()


@rag_router.post("/rag/knowledge/query_all")
def query_all_knowledge(user: UserNameRequest):
    """查询所有知识"""
    LogUtils.log_info("query_all_knowledge :", user.user_name)
    # return knowledge_dao.get_all_knowledges_by_user_name(user.user_name)
    return knowledge_dao.get_all_knowledges()


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
async def send_streaming_data(chat_request_data, parse_question, websocket, all_nodes):
    """用解析之后的问题去问大模型
    聊天记录也保存解析后的,后续解析更好的理解意图,不能用原始的问题,不然一直迭代,问题会越来越偏
    """
    start_time = time.time()
    # 4.1 send generate start flag ,not need ,auto

    # 4.2 send stream start flag
    await websocket.send_text(CHAT_STREAM_SERVE_START)
    await asyncio.sleep(0)

    # 4.3 generate stream
    stream_response = rag_base_manager.generate_chat_stream_response(
        parse_question, all_nodes
    )
    final_result = ""
    try:
        for chunk in stream_response.response_gen:
            if chunk is not None:
                final_result += str(chunk)
                await websocket.send_text(str(chunk))
            await asyncio.sleep(0)  # 让出控制权

    finally:
        inputs = [{"role": "user", "content": parse_question}]
        chat_manager.add_chat_record(
            chat_request_data.user_name, chat_request_data.session_id, inputs
        )

        outputs = [
            {
                "role": "assistant",
                "content": final_result,
                "model_name": "",
            }
        ]
        chat_manager.add_chat_record(
            chat_request_data.user_name, chat_request_data.session_id, outputs
        )

        follow_questions = generate_follow_questions(parse_question, final_result)
        LogUtils.log_info(f"end_stream_time: {time.time() - start_time}")

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(CHAT_STREAM_SERVE_DONE)
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

    rag_config = RagFrontendConfig(**chat_request_data.model_dump())
    LogUtils.log_info("rag_config =", rag_config)

    # 0.1 send parse question start
    await websocket.send_text(RAG_PARSE_QUESTION_START)
    await asyncio.sleep(0)
    if chat_request_data.rag_multi_turn_chat_enabled:
        history = get_chat_history(
            chat_request_data.user_name, chat_request_data.session_id
        )
        parse_question, parse_cost_time = rag_base_manager.parse_context_question(
            origin_query=chat_request_data.data, context=history
        )

    else:
        parse_question = chat_request_data.data
        parse_cost_time = "0"

    # 0.2 send parse question done
    # await websocket.send_text(RAG_PARSE_QUESTION_DONE + parse_cost_time)
    # await asyncio.sleep(0)
    parse_send_text = parse_question + RAG_PARSE_QUESTION_DONE + parse_cost_time
    await websocket.send_text(
        json.dumps({"rag_parse_context_question": parse_send_text})
    )

    # 1.1 send retrieve start flag
    # await websocket.send_text(RAG_RETRIEVE_CHUNK_START)
    # await asyncio.sleep(0)

    # 1.2 retrieve nodes
    retrieve_nodes, retrieve_cost_time = await rag_base_manager.aretrieve_chunk(
        query=parse_question, rag_config=rag_config
    )

    # 1.3 retrieve done
    await websocket.send_text(RAG_RETRIEVE_CHUNK_DONE + retrieve_cost_time)
    await asyncio.sleep(0)

    # 2.1 send rerank start flag, not need, when retrieve done,auto rerank

    # 2.2 rerank nodes
    rerank_nodes, rerank_cost_time = rag_base_manager.rerank_chunks(
        origin_query=parse_question,
        retrieve_nodes=retrieve_nodes,
        rag_config=rag_config,
    )
    # 2.3 rerank done
    await websocket.send_text(RAG_RERANK_CHUNK_DONE + rerank_cost_time)
    await asyncio.sleep(0)

    # 2.4 send rerank nodes to frontend
    nodes_payload = FrontendNodesPayload(
        chunk_frontend_nodes=cast_nodes_to_frontend(rerank_nodes)
    )
    nodes_payload_json = json.dumps(
        nodes_payload.__dict__, default=lambda o: o.__dict__
    )
    await websocket.send_text(nodes_payload_json)
    await asyncio.sleep(0)

    # 3.1 check image
    text_nodes, image_nodes = check_image_node(rerank_nodes)
    if image_nodes:
        # 3.2 send image_qa flag
        await websocket.send_text(RAG_EVENT_IMAGE_QA_START)
        await asyncio.sleep(0)

        # 3.3 modul llm generate
        image_nodes, image_qa_cost_time = (
            await rag_base_manager.agenerate_image_nodes_response(
                parse_question, image_nodes
            )
        )
        # 3.4 image_qa done
        await websocket.send_text(RAG_EVENT_IMAGE_QA_DONE + image_qa_cost_time)
        await asyncio.sleep(0)

    all_nodes = text_nodes + image_nodes

    # 4 generate final answer

    send_task = asyncio.create_task(
        send_streaming_data(chat_request_data, parse_question, websocket, all_nodes)
    )

    while not send_task.done():
        try:
            # LogUtils.log_info("waiting for message")
            request_msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            LogUtils.log_info("receive new text: ", request_msg)
            if CHAT_STREAM_CLIENT_STOP in request_msg:
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
                await websocket.send_text(CHAT_STREAM_SERVE_DONE)

    except Exception as e:
        LogUtils.log_error(f"Error: {e}")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError as e:
                LogUtils.log_error(f"Error while closing websocket: {e}")
        LogUtils.log_info("Connection closed")
