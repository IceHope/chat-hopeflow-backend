import json
from typing import List
from llama_index.core.schema import NodeWithScore
from utils.image_utils import get_image_base64, is_image_node


class FrontendNode:
    def __init__(
        self,
        node_id: str,
        text: str,
        file_id: str,
        score: str,
        file_type: str,
        file_path: str,
        file_name: str,
        image_base64: str,
    ):
        self.node_id = node_id
        self.text = text
        self.file_id = file_id
        self.score = score
        self.file_type = file_type
        self.file_path = file_path
        self.file_name = file_name
        self.image_base64 = image_base64


def cast_node_frontend(node_with_score: NodeWithScore) -> FrontendNode:
    node = node_with_score.node

    file_type = node.metadata.get("file_type", "")
    file_path = node.metadata.get("file_path", "")

    image_base64 = ""
    if is_image_node(file_type):
        image_base64 = get_image_base64(file_path)

    return FrontendNode(
        node_id=node.id_,
        text=node.text,
        file_id=node.metadata.get("file_id", ""),
        score=str(node_with_score.score),
        file_type=file_type,
        file_path=file_path,
        file_name=node.metadata.get("file_name", ""),
        image_base64=image_base64,
    )


def cast_nodes_to_frontend(nodes_with_score: List[NodeWithScore]) -> List[FrontendNode]:
    frontend_nodes = []
    for node in nodes_with_score:
        frontend_nodes.append(cast_node_frontend(node))
    return frontend_nodes


class FrontendNodesPayload:
    def __init__(
        self,
        chunk_frontend_nodes: list[FrontendNode],
    ):
        self.chunk_frontend_nodes = chunk_frontend_nodes


# 生成测试数据
def generate_test_data() -> None:
    # 直接创建FrontendNode测试数据
    test_frontend_node = FrontendNode(
        node_id="test_node_id",
        text="这是一个测试节点的文本内容。",
        file_id="test_file_001",
        score="0.85",
        file_type="txt",
        file_path="/path/to/test/file.txt",
    )

    print("测试FrontendNode:")
    print(f"node_id: {test_frontend_node.node_id}")
    print(f"text: {test_frontend_node.text}")
    print(f"file_id: {test_frontend_node.file_id}")
    print(f"score: {test_frontend_node.score}")
    print(f"file_type: {test_frontend_node.file_type}")
    print(f"file_path: {test_frontend_node.file_path}")

    # 创建多个FrontendNode测试数据
    test_frontend_nodes = [
        test_frontend_node,
        FrontendNode(
            node_id="test_node_id_2",
            text="这是第二个测试节点的文本内容。",
            file_id="test_file_002",
            score="0.75",
            file_type="pdf",
            file_path="/path/to/test/file2.pdf",
        ),
    ]

    # 测试FrontendNodesPayload类
    payload = FrontendNodesPayload(test_frontend_nodes)
    nodes_payload_json = json.dumps(payload.__dict__, default=lambda o: o.__dict__)
    print("\n测试FrontendNodesPayload类:")
    print(nodes_payload_json)
    print(f"chunk_frontend_nodes数量: {len(payload.chunk_frontend_nodes)}")
    for idx, node in enumerate(payload.chunk_frontend_nodes):
        print(f"\n节点 {idx + 1}:")
        print(f"node_id: {node.node_id}")
        print(f"text: {node.text}")
        print(f"file_id: {node.file_id}")
        print(f"score: {node.score}")
        print(f"file_type: {node.file_type}")
        print(f"file_path: {node.file_path}")


# 运行测试
if __name__ == "__main__":
    generate_test_data()
