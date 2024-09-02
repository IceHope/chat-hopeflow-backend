import time
import asyncio

from llama_index.core.schema import NodeWithScore

from utils.image_utils import get_image_base64_url
from utils.log_utils import LogUtils
from utils.multi_modal_utils import get_mutil_modal_config_model

IMAGE_QA_PROMPT = (
    "你是一个善于解析图片内容的专家,根据用户的问题,参考下面提供的照片,回答问题"
    "用户问题: {query}\n"
    "答复:\n"
)


class ImageNodeQAManager:
    def __init__(self):
        self.multi_modal_llm = get_mutil_modal_config_model()

    def generate_image_node_answer(self, query: str, image_nodes: list[NodeWithScore]):
        image_urls = []
        for image_node in image_nodes:
            image_urls.append(
                get_image_base64_url(image_node.node.metadata["file_path"])
            )

        inputs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": IMAGE_QA_PROMPT.format(query=query),
                    },
                ]
                + [
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in image_urls
                ],
            },
        ]
        start_time = time.time()

        try:
            response = self.multi_modal_llm.invoke(inputs).content

            cost_time = round(time.time() - start_time, 2)
            LogUtils.log_info(f"multi_modal_llm cost time: {cost_time} seconds")
            LogUtils.log_info(f"multi_modal_llm answer: {response}")
            return response
        except Exception as e:
            LogUtils.log_error("发生了一个错误：", str(e))
            return ""

    async def agenerate_image_node_answer(
        self, query: str, image_nodes: list[NodeWithScore]
    ):
        image_urls = []
        for image_node in image_nodes:
            image_urls.append(
                get_image_base64_url(image_node.node.metadata["file_path"])
            )

        inputs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": IMAGE_QA_PROMPT.format(query=query),
                    },
                ]
                + [
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in image_urls
                ],
            },
        ]
        start_time = time.time()

        try:
            response = await asyncio.to_thread(self.multi_modal_llm.invoke, inputs)
            response_content = response.content

            cost_time = round(time.time() - start_time, 2)
            LogUtils.log_info(f"async multi_modal_llm cost time: {cost_time} seconds")
            LogUtils.log_info(f"async multi_modal_llm answer: {response_content}")
            return response_content
        except Exception as e:
            LogUtils.log_error("异步处理中发生了一个错误：", str(e))
            return ""
