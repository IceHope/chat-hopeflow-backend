# 模型名称 单独输入价格 单独输出价格 (百万tokens)

# Groq
GROQ_NAMES = [
    ("llama-3.1-70b-versatile", 0.0, 0.0),
    ("llama-3.1-8b-instant", 0.0, 0.0),
    ("llama3-70b-8192", 0.0, 0.0),
    ("llama3-8b-8192", 0.0, 0.0),
    ("mixtral-8x7b-32768", 0.0, 0.0),
    ("gemma2-9b-it", 0.0, 0.0),
    ("gemma-7b-it", 0.0, 0.0),
]
# OpenAI
RATE = 7  # 汇率
OPENAI_NAMES = [
    ("gpt-4o", 5 * RATE, 15 * RATE),
    ("gpt-4o-2024-08-06", 2.5 * RATE, 10 * RATE),
    ("gpt-4o-2024-05-13", 5 * RATE, 15 * RATE),
    ("gpt-4o-mini", 0.15 * RATE, 0.6 * RATE),
    ("gpt-4o-mini-2024-07-18", 0.15 * RATE, 0.6 * RATE),
    ("gpt-3.5-turbo", 0.5 * RATE, 1.5 * RATE),
    ("o1-preview", 15 * RATE, 60 * RATE),
    ("o1-mini", 3 * RATE, 12 * RATE),
]


# 阿里通义千问,灵积平台
# https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-thousand-questions-metering-and-billing?spm=a2c4g.11186623.0.0.64785120Fcf37k
DASHSCOPE_NAMES = [
    ("qwen-max", 40, 120),
    ("qwen-max-longcontext", 40, 120),
    ("qwen-plus", 4, 12),
    ("qwen-turbo", 2, 6),
    ("qwen-long", 0.5, 2),
    ("qwen-vl-max", 20, 20),
    ("qwen-vl-max-0809", 20, 20),
    ("qwen-vl-plus", 8, 8),
]


# 硅基流动 开源部署
# https://docs.siliconflow.cn/docs/getting-started
SILICONFLOW_NAME = [
    ("meta-llama/Meta-Llama-3.1-405B-Instruct", 21, 21),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct", 4.13, 4.13),
    ("meta-llama/Meta-Llama-3-70B-Instruct", 4.13, 4.13),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", 0, 0),
    ("Pro/meta-llama/Meta-Llama-3.1-8B-Instruct", 0.42, 0.42),
    ("Qwen/Qwen2-72B-Instruct", 4.13, 4.13),
    ("Qwen/Qwen2-Math-72B-Instruct", 4.13, 4.13),
]

# 百度文心大,千帆平台
# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/hlrk4akp7#tokens%E7%94%A8%E9%87%8F%E5%90%8E%E4%BB%98%E8%B4%B9
QIANFAN_NAMES = [
    ("ERNIE-4.0-8K", 40, 120),
    ("ERNIE-4.0-8K-Latest", 40, 120),
    ("ERNIE-4.0-Turbo-8K", 30, 60),
    ("ERNIE-3.5-8K", 4, 12),
    ("ERNIE-3.5-128K", 8, 12),
    ("ERNIE-Speed-Pro-8K", 0.09, 0.2),
    ("ERNIE-Speed-Pro-128K", 0.18, 0.4),
    ("ERNIE-Speed-8K", 0, 0),
    ("ERNIE-Speed-128K", 0, 0),
    ("ERNIE-Lite-8K", 0, 0),
]
# 智谱
# https://open.bigmodel.cn/pricing
ZHIPU_NAMES = [
    ("GLM-4-Plus", 50, 50),
    ("GLM-4-0520", 100, 100),
    ("GLM-4-AirX", 10, 10),
    ("GLM-4-Air", 1, 1),
    ("GLM-4-Long", 1, 1),
    ("GLM-4-Flash", 0, 0),
    ("CodeGeeX-4", 0.1, 0.1),
    ("GLM-4V-Plus", 10, 10),
    ("GLM-4V", 50, 50),
    ("GLM-4-AllTools", 100, 100),
]


# DeepSeek
# https://platform.deepseek.com/api-docs/zh-cn/pricing/
DEEPSEEK_NAMES = [
    ("deepseek-chat", 1, 2),
    ("deepseek-coder", 1, 2),
]

# 百川
# https://platform.baichuan-ai.com/price
BAICHUAN_NAMES = [
    ("Baichuan4", 100, 100),
    ("Baichuan3-Turbo", 12, 12),
    ("Baichuan3-Turbo-128k", 24, 24),
    ("Baichuan2-Turbo", 8, 8),
    ("Baichuan2-Turbo-192k", 16, 16),
    ("Baichuan2-53B", 20, 20),  # 00:00 ~ 8:00的时间段价格10
]

# MinMax
# https://platform.minimaxi.com/document/Price?key=66701c7e1d57f38758d5818c
MINIMAX_NAMES = [
    ("abab6.5-chat", 30, 30),
    ("abab6.5s-chat", 10, 10),
    ("abab6.5t-chat", 5, 5),
    ("abab6.5g-chat", 5, 5),
    ("abab5.5-chat", 15, 15),
    ("abab5.5s-chat", 5, 5),
]
# Moonshot kimi
# https://platform.moonshot.cn/docs/price/chat#%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E4%BB%B7%E6%A0%BC%E8%AF%B4%E6%98%8E
MOONSHOT_NAMES = [
    ("moonshot-v1-8k", 12, 12),
    ("moonshot-v1-32k", 24, 24),
    ("moonshot-v1-128k", 60, 60),
]
# 讯飞星火 个人版免费
# https://console.xfyun.cn/sale/buy?wareId=9108&packageId=9108009&serviceName=Spark%20Max&businessId=bm35
SPARK_NAMES = [
    ("Spark4.0 Ultra", 0, 0),
    ("Spark Max", 0, 0),
    ("Spark Pro-128K", 0, 0),
    ("Spark Pro", 0, 0),
    ("Spark V2.0", 0, 0),
    ("Spark Lite", 0, 0),
]

# 零一万物
# https://platform.lingyiwanwu.com/docs
LINGYI_NAMES = [
    ("yi-large", 20, 20),
    ("yi-large-fc", 20, 20),
    ("yi-large-turbo", 12, 12),
    ("yi-medium", 2.5, 2.5),
    ("yi-medium-200k", 12, 12),
    ("yi-spark", 1, 1),
    ("yi-vision", 6, 6),
]

# 腾讯 混元
# https://cloud.tencent.com/document/product/1729/97731
HUANYUAN_NAMES = [
    ("hunyuan-pro", 30, 100),
    ("hunyuan-standard", 4.5, 5),
    ("hunyuan-standard-256k", 15, 60),
    ("hunyuan-lite", 0, 0),
]

# https://agicto.com/model
CLAUDE_NAMES = [
    ("claude-3-5-sonnet-20240620", 21.9, 109.5),
    ("claude-3-sonnet-20240229", 21.9, 109.5),
    ("claude-3-haiku-20240307", 1.83, 9.13),
    ("claude-3-opus-20240229", 109.5, 547.5),
]

CLAUDE_MULTIMODAL_NAMES = CLAUDE_NAMES

# https://agicto.com/model
MISTRAL_NAMES = [
    ("mixtral-8x7b-32768", 78.84, 78.84),
    ("mistral-large-latest", 449.68, 1349.04),
    ("mistral-medium-latest", 151.84, 455.52),
    ("mistral-small-latest", 112.42, 337.26),
    ("open-mixtral-8x7b", 39.42, 39.42),
    ("open-mistral-7b", 13.87, 13.87),
]
# https://agicto.com/model
GOOGLE_NAMES = [
    ("gemini-1.5-pro", 25.55, 76.65),
    ("gemini-1.5-flash", 0.55, 2.19),
    ("gemini-pro", 3.65, 10.95),
    ("gemma2-9b-it", 0, 0),
    ("gemma-7b-it", 0, 0),
]

# https://agicto.com/model
COHERE_NAMES = [
    ("command-light-nightly", 3.65, 10.95),
    ("command-light", 3.65, 10.95),
    ("command-nightly", 3.65, 10.95),
    ("command", 3.65, 10.95),
    ("command-r", 3.65, 10.95),
    ("command-r-plus", 21.9, 109.5),
]
# https://agicto.com/model
DOUBAO_NAMES = [
    ("Doubao-pro-128k", 5, 9),
    ("Doubao-pro-32k", 0.8, 2),
    ("Doubao-pro-4k", 0.8, 2),
    ("Doubao-lite-128k", 0.8, 1),
    ("Doubao-lite-32k", 0.3, 0.6),
    ("Doubao-lite-4k", 0.3, 0.6),
]

# https://agicto.com/model
META_NAMES = [
    ("Llama-3.1-405b", 36.5, 36.5),
    ("Llama-3.1-70b", 6.57, 6.57),
    ("Llama-3.1-8b", 1.46, 1.46),
    ("Llama-3-70b-chat-hf", 6.57, 6.57),
    ("Llama-3-8b-chat-hf", 1.46, 1.46),
    ("llama2-70b-4096", 10.95, 10.95),
    ("Llama-2-70b-chat-hf", 6.57, 6.57),
    ("Llama-2-13b-chat-hf", 2.19, 2.19),
    ("Llama-2-7b-chat-hf", 1.46, 1.46),
    ("CodeLlama-70b-Instruct-hf", 6.57, 6.57),
    ("CodeLlama-34b-Instruct-hf", 5.84, 5.84),
    ("CodeLlama-13b-Instruct-hf", 2.19, 2.19),
    ("CodeLlama-7b-Instruct-hf", 1.46, 1.46),
]
