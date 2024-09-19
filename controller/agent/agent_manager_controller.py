from fastapi import APIRouter

from controller.agent.agent_flow_desc import *
from schema.agent_schema import AgentConfigSchema, AgentItemId

agent_manager_router = APIRouter()


def get_translate_schema() -> AgentConfigSchema:
    return AgentConfigSchema(
        item_id=AgentItemId.TRANSLATE.value,
        title="三步翻译法",
        desc="通过直译,反思,意译实现精细化翻译,反思的过程,可以加入人类意见",
        frame="Langgraph",
        node_count=3,
        flow_desc=TRANSLATE_FLOW_DESC,
    )


def get_story_line_schema() -> AgentConfigSchema:
    return AgentConfigSchema(
        item_id=AgentItemId.STORY_LINE.value,
        title="故事创作",
        desc="""根据故事idea,按照顺序生成故事背景,情节大纲,和具体的一个个情节,
        每一个环节,都可以加入人类意见,可以反复迭代修改,也可以全自动生成,灵活配置""",
        frame="Langgraph",
        node_count=3,
        flow_desc=STORY_LINE_FLOW_DESC,
    )


def get_six_hat_schema() -> AgentConfigSchema:
    return AgentConfigSchema(
        item_id=AgentItemId.SIX_HAT.value,
        title="六顶思考帽",
        desc="六顶思维帽的思维模式,去协作讨论问题,下一步的思考可以参考上一步的内容",
        frame="Langgraph",
        node_count=6,
        flow_desc=SIX_HAT_FLOW_DESC,
    )


def get_openai_o1_schema() -> AgentConfigSchema:
    return AgentConfigSchema(
        item_id=AgentItemId.OPENAI_O1.value,
        title="推理模型",
        desc="OpenAI O1的推理模型体验",
        frame="Langgraph",
        node_count=1,
        flow_desc=OPENAI_O1_FLOW_DESC,
    )


@agent_manager_router.get("/agent/list_all")
async def get_command_config() -> list[AgentConfigSchema]:
    return [
        get_translate_schema(),
        get_story_line_schema(),
        get_six_hat_schema(),
        get_openai_o1_schema(),
    ]
