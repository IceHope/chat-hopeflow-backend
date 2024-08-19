from typing import Optional

from schema.chat_schema import ChatRequestData


class TranslationAgentSchema(ChatRequestData):
    source_lang: str
    target_lang: str
    source_text: str = None
    country: Optional[str] = None
    human_flag: bool = True


class StoryLineAgentSchema(ChatRequestData):
    human_flag: bool = True
