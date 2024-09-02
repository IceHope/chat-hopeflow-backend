import time
from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    TextNode,
)

from utils.log_utils import LogUtils


def build_nodes_from_documents_without_split(
        documents: List[BaseNode],
) -> List[TextNode]:
    """
    从文档构建节点,不进行分割

    Args:
        documents: 文档列表

    Returns:
        构建的节点列表
    """

    nodes: List[TextNode] = []

    for document in documents:
        if isinstance(document, ImageDocument):
            # 处理图像文档
            image_node = ImageNode(
                id_=document.id_,
                text=document.text,
                metadata=document.metadata,
                embedding=document.embedding,
                image=document.image,
                image_path=document.image_path,
                image_url=document.image_url,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
            )
            nodes.append(image_node)  # type: ignore
        elif isinstance(document, Document):
            # 处理文本文档
            node = TextNode(
                id_=document.id_,
                text=document.text,
                metadata=document.metadata,
                embedding=document.embedding,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
            )
            nodes.append(node)
        else:
            raise ValueError(f"未知的文档类型: {type(document)}")

    return nodes


class ChunkManager:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200) -> None:
        """
        初始化ChunkManager

        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def get_chunk_size(self) -> int:
        return self.chunk_size

    def get_chunk_overlap(self) -> int:
        return self.chunk_overlap

    def chunk_documents(self, documents: list[Document]) -> List[BaseNode]:
        """
        对文本文档进行分块

        Args:
            documents: 文档列表

        Returns:
            分块后的节点列表
        """
        start_time = time.time()

        text_documents = []
        image_documents = []
        for document in documents:
            if isinstance(document, ImageDocument):
                image_documents.append(document)
            else:
                text_documents.append(document)

        LogUtils.log_info(
            f"文本文档数量: {len(text_documents)}, 图像文档数量: {len(image_documents)}"
        )

        nodes = []
        if text_documents:
            text_nodes = self.node_parser.get_nodes_from_documents(
                text_documents, show_progress=True
            )
            nodes.extend(text_nodes)
            LogUtils.log_info(f"文本文档处理完成，生成了 {len(text_nodes)} 个节点")

        if image_documents:
            image_nodes = build_nodes_from_documents_without_split(image_documents)
            nodes.extend(image_nodes)
            LogUtils.log_info(f"图像文档处理完成，生成了 {len(image_nodes)} 个节点")

        LogUtils.log_info(f"总共生成了 {len(nodes)} 个节点")

        elapsed_time = round(time.time() - start_time, 2)

        LogUtils.log_info(f"分块耗时: {elapsed_time} 秒")

        return nodes
