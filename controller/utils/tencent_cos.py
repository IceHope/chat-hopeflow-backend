# -*- coding=utf-8
# https://cloud.tencent.com/document/product/436/65820
import os
import threading
import time
from datetime import datetime

from dotenv import load_dotenv
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

from config import PATH_PROJECT_IMAGES
from utils.log_utils import LogUtils

load_dotenv()

secret_id = os.getenv('HUNYUAN_SECRET_ID')
secret_key = os.getenv('HUNYUAN_SECRET_KEY')
region = 'ap-beijing'

config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
client = CosS3Client(config)

bucket = 'icehope-1326453681'


def get_tencent_cloud_url(image_path: str):
    LogUtils.log_info("current_thread: ", threading.current_thread().name)
    LogUtils.log_info("current_thread_id: ", threading.get_ident())
    key_name = "images/upload/" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S_') + os.path.basename(image_path)
    # 本地路径 简单上传
    response = client.put_object_from_local_file(
        Bucket=bucket,
        LocalFilePath=image_path,
        Key=key_name
    )
    time.sleep(10)
    url = client.get_object_url(
        Bucket=bucket,
        Key=key_name
    )
    print('get_tencent_cloud_url= ' + url)
    return url


if __name__ == '__main__':
    get_tencent_cloud_url(PATH_PROJECT_IMAGES + "/boy_1.jpg")