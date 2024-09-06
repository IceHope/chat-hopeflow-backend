from factory.embedding.dashscope_embedding import DashscopeEmbeddingFactory
from factory.embedding.open_bedding import OpenaiEmbeddingFactory
from factory.embedding.qianfan_embedding import QianfanEmbeddingFactory
from factory.embedding.zhipu_embedding_factory import ZhipuEmbeddingFactory
from factory.mode_type import EmbeddingType


class EmbeddingFactory:
    @staticmethod
    def get_embedding(embedding_type: EmbeddingType):
        if embedding_type == EmbeddingType.OPENAI:
            return OpenaiEmbeddingFactory().get_embedding()

        if embedding_type == EmbeddingType.DASHSCOPE:
            return DashscopeEmbeddingFactory().get_embedding()

        if embedding_type == EmbeddingType.QIANFAN:
            return QianfanEmbeddingFactory().get_embedding()

        if embedding_type == EmbeddingType.ZHIPU:
            return ZhipuEmbeddingFactory().get_embedding()


if __name__ == "__main__":
    embedding_mode = EmbeddingFactory.get_embedding(EmbeddingType.DASHSCOPE)
    embedding = embedding_mode.embed_query("What is the capital of France?")
    print(len(embedding))
