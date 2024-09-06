import toml
from pathlib import Path


class RagConfiguration:
    """
    Base class for configuration
    """

    def __init__(self):
        path = str(Path(__file__).parent / "rag_settings.toml")
        # 打开并读取 TOML 文件
        with open(path, 'r', encoding="utf-8") as file:
            self.config = toml.load(file)

    def get_multi_modal_config(self):
        return self.config['rag']['multi_modal']["source"]

    def get_milvus_uri(self):
        return self.config['rag']['milvus']['uri']

    def get_embedding_config(self):
        return self.config['rag']['embedding']['config']

    def get_embedding_name(self):
        return self.config['rag']['embedding']['config']['config_model_name']

    def get_embedding_info(self, simple_embedding_name: str):
        return self.config['rag']['embedding'][simple_embedding_name]

    def get_retriever_config(self):
        return self.config['rag']['retriver']

    def get_similarity_top_k(self):
        return self.config['rag']['retriver']['similarity_top_k']

    def get_rerank_top_n(self):
        return self.config['rag']['retriver']['rerank_top_n']


if __name__ == '__main__':
    config = RagConfiguration().get_embedding_config()
    print(config["type"])
