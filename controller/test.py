import asyncio

import inspect

import uuid


async def task1():
    print(f"task1 coroutine : {inspect.stack()[0].function}")
    print(f"task1 uuid4 : {uuid.uuid4()}")
    print("Task 1 starts")
    await asyncio.sleep(3)  # 模拟 I/O 操作，非阻塞等待
    print("Task 1 finishes")


async def task2():
    print(f"task2 coroutine : {inspect.stack()[0].function}")
    print(f"task2 uuid4 : {uuid.uuid4()}")
    print("Task 2 starts")
    await asyncio.sleep(6)  # 模拟另一个 I/O 操作，非阻塞等待
    print("Task 2 finishes")


async def main():
    print(f"coroutine : {inspect.stack()[0].function}", f" uuid4 : {uuid.uuid4()}")

    # 创建协程对象，但尚未运行
    t1 = task1()
    t2 = task2()

    # 并发运行两个协程
    await asyncio.gather(t1, t2)

    print("All tasks finished")


for i in range(2):
    asyncio.run(main())
