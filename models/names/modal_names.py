from models.names.llm_names import OPENAI_NAMES, CLAUDE_NAMES

OPENAI_MULTIMODAL_NAMES = OPENAI_NAMES[:-1]

CLAUDE_MULTIMODAL_NAMES = CLAUDE_NAMES

GOOGLE_MULTIMODAL_NAMES = [
    ("gemini-1.5-pro", 25.55, 76.65),
    ("gemini-1.5-flash", 0.55, 2.19),
]

DASHSCOPE_MULTIMODAL_NAMES = [
    ("qwen-vl-max", 20, 20),
    ("qwen-vl-max-0809", 20, 20),
    ("qwen-vl-plus", 8, 8),
]

ZHIPU_MULTIMODAL_NAMES = [
    ("GLM-4V-Plus", 10, 10),
    ("GLM-4V", 50, 50),
]

LINGYI_MULTIMODAL_NAMES = [
    ("yi-vision", 6, 6),
]