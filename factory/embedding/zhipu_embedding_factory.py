import os

from factory.embedding.base_embedding import BaseEmbeddingFactory


class ZhipuEmbeddingFactory(BaseEmbeddingFactory):
    def get_embedding(self):
        from langchain_community.embeddings import ZhipuAIEmbeddings

        return ZhipuAIEmbeddings(
            api_key=os.getenv("ZHIPUAI_API_KEY"),
        )


if __name__ == "__main__":
    embedding = ZhipuEmbeddingFactory().get_embedding().embed_query(
        "This is a test query.")
    print(len(embedding))
