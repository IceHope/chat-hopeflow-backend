import threading

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os

from config import PATH_PROJECT_IMAGES
from controller.utils.tencent_cos import get_tencent_cloud_url
from utils.log_utils import LogUtils

# 创建一个文件上传的路由
file_router = APIRouter()

# 定义文件保存的临时目录
TEMP_IMAGE_DIR = os.path.join(PATH_PROJECT_IMAGES, "tem")


@file_router.post("/images/upload")
async def upload_image(file: UploadFile = File(...)):
    LogUtils.log_info("current_thread: ", threading.current_thread().name)
    LogUtils.log_info("current_thread_id: ", threading.get_ident())
    try:
        # 确保临时目录存在
        os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

        # 生成文件的临时路径
        temp_image_path = os.path.join(TEMP_IMAGE_DIR, file.filename)

        # 将文件保存到临时目录
        with open(temp_image_path, "wb") as f:
            f.write(await file.read())

        # 上传到腾讯云并获取URL
        image_url = get_tencent_cloud_url(temp_image_path)

        # 删除本地的临时文件
        os.remove(temp_image_path)

        return JSONResponse(content={"url": image_url, "message": "文件上传成功"})
    except Exception as e:
        # 错误处理
        print(e)
        return JSONResponse(status_code=500, content={"message": str(e)})
