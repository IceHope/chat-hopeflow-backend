from factory.embedding.base_embedding import BaseEmbeddingFactory
import os


class DashscopeEmbeddingFactory(BaseEmbeddingFactory):
    def get_embedding(self):
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="text-embedding-v2"
        )


if __name__ == "__main__":
    embedding = DashscopeEmbeddingFactory().get_embedding().embed_query(
        "This is a test query.")
    print(len(embedding))
