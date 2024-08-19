from typing import List

from FlagEmbedding import BGEM3FlagModel
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction


class ExampleSparseEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self):
        model_dir = "F:/HuggingFace/Embedding/bge-m3"
        self.model = BGEM3FlagModel(model_dir, use_fp16=False)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result
