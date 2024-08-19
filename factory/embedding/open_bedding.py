from factory.embedding.base_embedding import BaseEmbeddingFactory
import os


class OpenaiEmbeddingFactory(BaseEmbeddingFactory):
    def get_embedding(self):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            dimensions=512,
        )


if __name__ == "__main__":
    embedding = OpenaiEmbeddingFactory().get_embedding().embed_query(
        "This is a test query.")
    print(len(embedding))
