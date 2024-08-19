import time
import uuid

# print(uuid.uuid4())

print(time.time())
print(int(time.time()))

for i in range(10):
    time.sleep(1)
    print(int(time.time()))



