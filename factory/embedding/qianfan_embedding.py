import os

from factory.embedding.base_embedding import BaseEmbeddingFactory


class QianfanEmbeddingFactory(BaseEmbeddingFactory):
    def get_embedding(self):
        from langchain_community.embeddings import QianfanEmbeddingsEndpoint

        # Embedding-V1   384
        # bge-large-en   1024
        # bge-large-zh   1024

        return QianfanEmbeddingsEndpoint(
            qianfan_ak=os.getenv("ERNIE_CLIENT_ID"),
            qianfan_sk=os.getenv("ERNIE_CLIENT_SECRET"),
            model="tao-8k"
        )


if __name__ == "__main__":
    embedding = QianfanEmbeddingFactory().get_embedding().embed_query(
        "This is a test query.")
    print(len(embedding))
