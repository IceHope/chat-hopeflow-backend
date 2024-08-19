from openai import OpenAI
# https://github.com/ultrasev/llmproxy

# api_key = "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"
# model = "llama3-8b-8192"
# BASE_URL = "https://llmapi.ultrasev.com/v2/groq"

api_key = "AIzaSyDNzHTqtM1dWFsfJ8AoEiferNbmgHL1F_4"
# model = "gemini-1.5-flash"
model = "gemini-1.5-pro"
BASE_URL = "https://llmapi.ultrasev.com/v2/gemini"

client = OpenAI(base_url=BASE_URL, api_key=api_key)

image_url="https://icehope-1326453681.cos.ap-beijing.myqcloud.com/images/upload/2024-08-04-11-16-23-boy_1.jpg"
inputs = [
    {"role": "user",
     "content": [
         {"type": "text", "text": "这张图片描述了什么"},
         {"type": "image_url", "image_url": {"url": image_url}},
     ]},
]

# response = client.chat.completions.create(
#     model=model,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant。"},
#         {"role": "user", "content": "what is the meaning of life?"}
#     ],
#     stream=False
# )

response = client.chat.completions.create(
    model=model,
    messages=inputs,
    stream=False
)

# print(response.choices[0].message.content)
print(response)
# for chunk in response:
#     print(chunk.choices[0].delta.content, end="")