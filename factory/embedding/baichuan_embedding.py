import os

from factory.embedding.base_embedding import BaseEmbeddingFactory


class BaichuanEmbeddingFactory(BaseEmbeddingFactory):
    def get_embedding(self):
        from langchain_community.embeddings import BaichuanTextEmbeddings

        return BaichuanTextEmbeddings(
            baichuan_api_key=os.getenv("BAICHUAN_API_KEY"),
        )


if __name__ == "__main__":
    embedding = BaichuanEmbeddingFactory().get_embedding().embed_query(
        "This is a test query.")
    print(len(embedding))
