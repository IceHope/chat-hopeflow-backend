from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()


class BaseEmbeddingFactory(ABC):
    @abstractmethod
    def get_embedding(self):
        pass
