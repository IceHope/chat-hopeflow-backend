from enum import Enum
from typing import Optional

from pydantic import BaseModel

from schema.chat_schema import ChatRequestData


class TranslationAgentSchema(ChatRequestData):
    source_lang: str
    target_lang: str
    source_text: str = None
    country: Optional[str] = None
    human_flag: bool = True


class StoryLineAgentSchema(ChatRequestData):
    background_human_flag: bool = False
    storyline_human_flag: bool = False
    detail_human_flag: bool = False


class AgentItemId(str, Enum):
    TRANSLATE = "agent_translate"
    STORY_LINE = "agent_storyline"
    SIX_HAT = "agent_six_hat"
    OPENAI_O1 = "agent_openai_o1"


class AgentConfigSchema(BaseModel):
    item_id: str
    title: str
    desc: str
    frame: str
    node_count: int
    flow_desc: str
