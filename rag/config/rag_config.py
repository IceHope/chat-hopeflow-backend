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

    def get_bge_embedding_config(self):
        return self.config['rag']['embedding']['bge']['source']

    def get_bge_embedding_m3_path(self):
        return self.config['rag']['embedding']['bge-m3']['file_path']

    def get_similarity_top_k(self):
        return self.config['rag']['retriver']['similarity_top_k']

    def get_rerank_top_n(self):
        return self.config['rag']['retriver']['rerank_top_n']


if __name__ == '__main__':
    config = RagConfiguration().get_rerank_top_n()
    print(config)
