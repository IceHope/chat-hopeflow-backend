import json
import time

import redis

# 连接到Redis服务器
r = redis.Redis(host='localhost', port=6379, db=0)


# 增加数据
def add_data(key, value):
    r.set(key, value)
    print(f"Added: {key} -> {value}")


# 查询数据
def get_data(key):
    value = r.get(key)
    if value:
        print(f"Fetched: {key} -> {value.decode('utf-8')}")
    else:
        print(f"No data found for key: {key}")


# 修改数据
def update_data(key, new_value):
    add_data(key, new_value)  # SET命令同时用于增加和修改


# 删除数据
def delete_data(key):
    if r.delete(key):
        print(f"Deleted key: {key}")
    else:
        print(f"Key not found: {key}")


# 主程序
if __name__ == "__main__":
    # # 添加数据示例
    # add_data("username2", "kimi")
    # add_data("age2", "104")
    # #
    # # # 查询数据示例
    # get_data("username2")
    # #
    # # 修改数据示例
    # update_data("age", "101")
    #
    # # 再次查询以查看更新结果
    # get_data("age")
    #
    # # 删除数据示例
    # delete_data("username")
    #
    # # 尝试查询已删除的键
    # get_data("username")
    # r.select(5)

    # 初始聊天记录
    messages = [
        {"role": "user1", "content": "介绍下你自己,版本号,生产商"},
        {"role": "assistant2", "content": "我是一个AI助手,我的版本号是1.0.6,我是由Lingyi团队开发的"},
        {"role": "user3", "content": "我的名字是kimi"},
        {"role": "assistant4", "content": "你好kimi"},
        {"role": "user5", "content": "你是谁"},
    ]

    # 将聊天记录添加到有序集合
    for index, message in enumerate(messages):
        r.zadd('bill:chat', {json.dumps(message): time.time()})
