from typing import List, Any

from llama_index.core.schema import BaseNode
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import MILVUS_ID_FIELD, logger


class PartitionMilvusVectorStore(MilvusVectorStore):
    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add the embeddings and their nodes into Milvus.

        Args:
            nodes (List[BaseNode]): List of nodes with embeddings
                to insert.

        Raises:
            MilvusException: Failed to insert data.

        Returns:
            List[str]: List of ids inserted.
        """
        insert_list = []
        insert_ids = []

        if self.enable_sparse is True and self.sparse_embedding_function is None:
            logger.fatal(
                "sparse_embedding_function is None when enable_sparse is True."
            )

        # Process that data we are going to insert
        for node in nodes:
            entry = node_to_metadata_dict(node)
            entry[MILVUS_ID_FIELD] = node.node_id
            entry[self.embedding_field] = node.embedding

            if self.enable_sparse is True:
                entry[
                    self.sparse_embedding_field
                ] = self.sparse_embedding_function.encode_documents([node.text])[0]

            insert_ids.append(node.node_id)
            insert_list.append(entry)

        # Insert the data into milvus
        print("Insert the data into milvus")
        for insert_batch in iter_batch(insert_list, self.batch_size):
            self._collection.insert(data=insert_batch, partition_name="partitionC")
            # self._collection.insert(data=insert_batch)
        if add_kwargs.get("force_flush", False):
            self._collection.flush()
        self._create_index_if_required()
        logger.debug(
            f"Successfully inserted embeddings into: {self.collection_name} "
            f"Num Inserted: {len(insert_list)}"
        )
        return insert_ids
