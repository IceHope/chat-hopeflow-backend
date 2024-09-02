from collections import defaultdict
from typing import List

from FlagEmbedding import BGEM3FlagModel
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction

from rag.config.rag_config import RagConfiguration
from utils.log_utils import LogUtils


class LocalSparseEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self):
        bge_m3_dir = RagConfiguration().get_bge_embedding_m3_path()
        LogUtils.log_info("Loading BGE model from: {}".format(bge_m3_dir))
        self.model = BGEM3FlagModel(bge_m3_dir, use_fp16=False)

    def _encode_retry(self, error_text: str):
        # 重试,目前重试基本都失败
        retry_output = self.model.encode(
            error_text,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        LogUtils.log_info(f"retry output: {retry_output}")

        if retry_output is None or len(retry_output) == 0:
            # 异常数据的清洗标准化,保证在保存milvus的时候,不会出错
            # "hello World" 的向量代替
            retry_output = defaultdict(int, {'33600': 1.2171164, '31': 1.0892353, '6661': 1.2170767})
        return retry_output

    def encode_queries(self, queries: List[str]):
        query_outputs = self.model.encode(
            queries,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        LogUtils.log_info("BGEM3FlagModel encode_queries : --- ", queries)
        LogUtils.log_info(
            "BGEM3FlagModel sparse_embedding : --- ",
            self.model.convert_id_to_token(query_outputs),
        )
        return [self._to_standard_dict(output) for output in query_outputs]

    def encode_documents(self, origin_documents: List[str]):
        doc_outputs = self.model.encode(
            origin_documents,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        for i, item_output in enumerate(doc_outputs):
            # 1.如果是单个字母或者其他字符,经常会返回空,
            # 2.有时候不稳定,整行的句子也会返回空
            # 然后添加到milvus数据库的时候,会报错,数据格式错误
            if item_output is None or len(item_output) == 0:
                LogUtils.log_error("BGEM3FlagModel encode_documents error: --- ", origin_documents[i])
                LogUtils.log_error("BGEM3FlagModel sparse_embedding error: --- ", item_output)

                # doc_outputs[i] = self._encode_retry(origin_documents[i])
                doc_outputs[i] = defaultdict(int, {'33600': 1.2171164, '31': 1.0892353, '6661': 1.2170767})

        return [self._to_standard_dict(output) for output in doc_outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result


if __name__ == "__main__":
    model_dir = "F:/HuggingFace/Embedding/bge-m3"
    model = BGEM3FlagModel(model_dir, use_fp16=False)
    documents: List[str] = ["hello world", "hello milvus"]
    outputs = model.encode(
        documents,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )["lexical_weights"]
