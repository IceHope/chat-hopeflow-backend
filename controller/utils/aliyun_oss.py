import os

import oss2
from dotenv import load_dotenv

from config import PATH_PROJECT_IMAGES

load_dotenv()

# 使用获取的RAM用户的访问密钥配置访问凭证
auth = oss2.AuthV4(
    access_key_id=os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'),
    access_key_secret=os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'))

# 填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
endpoint = 'https://oss-cn-beijing.aliyuncs.com'
# 填写Endpoint对应的Region信息，例如cn-hangzhou。
region = 'cn-beijing'
# 填写Bucket名称。
bucket = oss2.Bucket(auth=auth, endpoint=endpoint, bucket_name='icehope', region=region, connect_timeout=16)


def get_cloud_url(image_path: str):
    cloud_file_name = "images/tem/" + os.path.split(image_path)[1]

    bucket.put_object_from_file(cloud_file_name, image_path)
    url = bucket.sign_url('GET', cloud_file_name, 5 * 60)
    return url


if __name__ == '__main__':
    path = PATH_PROJECT_IMAGES + "/boy_1.jpg"
    url = get_cloud_url(path)
    print(url)