import json
import random
import time
from datetime import datetime

import redis
from pydantic import BaseModel

from config.base_config import BaseConfiguration
from utils.log_utils import LogUtils


class ChatHistoryModel(BaseModel):
    user_question: str
    ai_reply: str
    summary: str


class ChatRedisManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        redis_config = BaseConfiguration().get_redis_config()
        self.redis_client = redis.Redis(
            host=redis_config["redis_host"],
            port=redis_config["redis_port"],
            decode_responses=True
        )
        self.redis_client.execute_command('SELECT', 2)

    def add_chat_record(self, username, sessionid, msg_list):
        key = f"{username}:{sessionid}"
        LogUtils.log_info("add_chat_record: ", key)
        for msg in msg_list:
            timestamp = time.time()
            self.redis_client.zadd(key, {json.dumps(msg): timestamp})

    def get_history_snapshots(self, username):
        keys = self.redis_client.keys(f"{username}:*")
        # 按照时间戳从大到小进行排序
        sorted_keys = sorted(keys, key=lambda k: int(k.split(":")[1]), reverse=True)
        LogUtils.log_info("sorted_keys :", sorted_keys)
        snapshots = []  # 使用列表来存储快照
        for key in sorted_keys:
            messages = self.redis_client.zrange(key, 0, -1)  # 获取所有消息
            messages = [json.loads(msg) for msg in messages]
            if messages:
                last_message = messages[-2] if len(messages) >= 2 else messages[-1]
                content = last_message["content"]

                if isinstance(content, list):
                    # 提取多模态图片提问的text
                    content = content[0]["text"]

                session_id = key.split(":")[1]
                snapshot = {
                    "user_name": username,
                    "session_id": int(session_id),  # 转换为整数以保持一致性
                    "last_msg": content
                }
                snapshots.append(snapshot)  # 将快照添加到列表中
        LogUtils.log_info("snapshots: ", snapshots)
        return snapshots

    def get_history_record(self, username, sessionid):
        key = f"{username}:{sessionid}"
        LogUtils.log_info("get_history_record ", key)
        if self.redis_client.exists(key):
            messages = self.redis_client.zrange(key, 0, -1)
            return [json.loads(msg) for msg in messages]
        else:
            return None

    def delete_chat_record(self, username, sessionid):
        key = f"{username}:{sessionid}"
        if self.redis_client.exists(key):
            self.redis_client.delete(key)


def generate_random_message():
    roles = ['user', 'assistant']
    return {
        "role": random.choice(roles),
        "content": f"随机消息内容 {datetime.now().isoformat()}"
    }


def generate_test_data(num_users=3, num_sessions_per_user=5, num_messages_per_session=10):
    manager = ChatRedisManager()
    manager.redis_client.execute_command('SELECT', 3)
    manager.redis_client.flushdb()
    for i in range(num_users):
        username = f"hope{i}"
        for j in range(num_sessions_per_user):
            time.sleep(1)
            msg_list = [generate_random_message() for _ in range(num_messages_per_session)]
            manager.add_chat_record(username=username, sessionid=str(int(time.time())), msg_list=msg_list)


# Example usage:
if __name__ == "__main__":
    generate_test_data()
    # manager = ChatRedisManager()
    # manager.redis_client.execute_command('SELECT', 3)
    # manager.redis_client.flushdb()
    # snapshots = manager.get_history_snapshots("QWE")
    # print(snapshots)
    # print("-------------")
    # all_keys = manager.redis_client.keys("*")
    # print("all_keys: ", all_keys)
    # records = json.loads(manager.redis_client.get(all_keys[0]))
    # pprint(type(records))
    # pprint(records)
