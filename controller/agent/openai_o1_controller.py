import json
import os
from datetime import datetime

from fastapi import APIRouter
from starlette.websockets import WebSocket

from models.factory.llm_factory import LLMFactory
from models.model_type import LLMType
from schema.chat_schema import ChatRequestData
from utils.command_constants import CHAT_STREAM_SERVE_DONE
from openai import OpenAI
from models.api_key_config import get_openai_config

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Openai_o1" + datetime.now().strftime(
    "%d/%m/%Y %H:%M:%S"
)

openai_o1_router = APIRouter()


@openai_o1_router.websocket("/ws/agent/openai_o1_reason")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        print("connection open")
        while True:
            try:
                request_msg = await websocket.receive_text()
                print(f"Received message: {request_msg}")

                chat_request_data = ChatRequestData(**json.loads(request_msg))

                _config = get_openai_config()

                client = OpenAI(
                    api_key=_config.api_key,
                    base_url=_config.base_url,
                )
                response = client.chat.completions.create(
                    model=chat_request_data.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": chat_request_data.data},
                            ],
                        }
                    ],
                )
                text = response.choices[0].message.content

                print("text: ", text)
                await websocket.send_text(text)
                await websocket.send_text(CHAT_STREAM_SERVE_DONE)

            except Exception as e:
                print(f"Error while receiving or processing data: {e}")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing connection")
        await websocket.close()
