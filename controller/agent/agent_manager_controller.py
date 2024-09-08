from fastapi import APIRouter

from controller.agent.agent_flow_desc import *
from schema.agent_schema import AgentConfigSchema, AgentItemId

agent_manager_router = APIRouter()


def get_translate_schema():
    return AgentConfigSchema(
        item_id=AgentItemId.TRANSLATE.value,
        title="三步翻译法",
        desc="通过直译,反思,意译实现精细化翻译",
        node_count=3,
        flow_desc=TRANSLATE_FLOW_DESC,
    )


def get_story_line_schema():
    return AgentConfigSchema(
        item_id=AgentItemId.STORY_LINE.value,
        title="故事创作",
        desc="背景,大纲,情节反复迭代生成故事",
        node_count=4,
        flow_desc=STORY_LINE_FLOW_DESC,
    )


@agent_manager_router.get("/agent/list_all")
async def get_command_config():
    return [get_translate_schema(), get_story_line_schema()]
