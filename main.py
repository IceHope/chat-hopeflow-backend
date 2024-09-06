import logging
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from controller.agent.story_line_controller import storyline_router
from controller.agent.translate_human_controller import translate_human_router
from controller.chat_controller import chat_router
from controller.config.config_controller import config_router
from controller.file_controller import file_router
from controller.rag.rag_controller import rag_router
from controller.user_controller import user_router

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(chat_router)
app.include_router(translate_human_router)
app.include_router(storyline_router)
app.include_router(file_router)
app.include_router(user_router)
app.include_router(rag_router)
app.include_router(config_router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8585)
