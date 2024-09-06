from openai import OpenAI


def get_image_base64_url(image_path: str, quality=85):
    import base64
    # 打开图片文件
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    print(len(img_base64))
    return f"data:image/jpeg;base64,{img_base64}"


# https://github.com/ultrasev/llmproxy

# api_key = "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"
# model = "llama3-8b-8192"
# BASE_URL = "https://llmapi.ultrasev.com/v2/groq"

api_key = "AIzaSyDNzHTqtM1dWFsfJ8AoEiferNbmgHL1F_4"
# model = "gemini-1.5-flash"
model = "gemini-1.5-pro"
BASE_URL = "https://llmapi.ultrasev.com/v2/gemini"

client = OpenAI(base_url=BASE_URL, api_key=api_key)
# base64 = get_image_base64_url("math.png")
# inputs = [
#     {"role": "user",
#      "content": [
#          {"type": "text", "text": "这张图片描述了什么"},
#          {"type": "image_url", "image_url": {"url": base64}},
#      ]},
# ]
inputs = [
    {"role": "system", "content": "You are a helpful assistant。"},
    {"role": "user", "content": "你是谁"}
]
response = client.chat.completions.create(
    model=model,
    messages=inputs,
    stream=False
)

# print(response.choices[0].message.content)
print(response)
# for chunk in response:
#     print(chunk.choices[0].delta.content, end="")
