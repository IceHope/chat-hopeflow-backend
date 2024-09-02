import json
import pprint
from typing import List, Optional

from pydantic import BaseModel
from pymilvus import MilvusClient

from rag.config.rag_config import RagConfiguration
from utils.image_utils import get_image_base64
from utils.log_utils import LogUtils


# 跟文件关联的节点
class NodeFromFilePath(BaseModel):
    file_id: str
    file_path: str
    file_type: str
    total_pages: Optional[int] = None
    source: Optional[int] = None
    node_id: str
    text: str
    image_base64: Optional[str] = None


def _parse_node_content(data: dict) -> NodeFromFilePath:
    # 解析 _node_content 字符串
    node_content = json.loads(data["_node_content"])

    # 从 node_content 中提取 text
    text = node_content.get("text", "")
    node_id = node_content.get("id_", "")

    # 从原始数据中提取其他字段
    file_id = data.get("file_id", "")
    file_path = data.get("file_path", "")
    file_type = data.get("file_type", "")
    total_pages = data.get("total_pages")
    source = data.get("source")
    image_base64 = None
    if file_type == "image/png":
        image_base64 = get_image_base64(file_path)

    # 创建并返回 NodesFromFilePath 对象
    return NodeFromFilePath(
        file_id=file_id,
        file_path=file_path,
        file_type=file_type,
        total_pages=total_pages,
        source=source,
        node_id=node_id,
        text=text,
        image_base64=image_base64,
    )


class MyMilvusClient:

    def __init__(self):
        self.client = MilvusClient(uri=RagConfiguration().get_milvus_uri())

    def search_nodes_from_file_id(self, collect_name, file_id: str):
        # 构建 SQL 过滤条件
        filter_condition = f'file_id == "{file_id}"'
        print(filter_condition)
        query_list = self.client.query(
            collection_name=collect_name,
            filter=filter_condition,
            output_fields=[
                "file_id",
                "file_path",
                "file_type",
                "total_pages",
                "source",
                "_node_content",
            ],
        )
        nodes: List[NodeFromFilePath] = []
        for item in query_list:
            nodes.append(_parse_node_content(item))

        # 对节点进行排序，按照source属性从小到大排列，None值排在最后
        nodes.sort(key=lambda x: (x.source is None, x.source))

        return nodes

    def delete_nodes_from_file_id(self, collect_name, file_id: str):
        filter_condition = f'file_id == "{file_id}"'
        res = self.client.delete(
            collection_name=collect_name,
            filter=filter_condition
        )
        LogUtils.log_info(f"delete nodes from {collect_name} with file_id {file_id} res: {res}")


if __name__ == "__main__":
    client = MyMilvusClient()
    file_id = "35732709-653d-4bb5-9355-4f18db14e000"
    nodes = client.search_nodes_from_file_id(
        collect_name="test_frontend", file_id=file_id
    )
    for node in nodes:
        pprint.pprint(node)
