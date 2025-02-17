from abc import ABC, abstractmethod
from pathlib import Path


class BaseChatbot(ABC):
    @abstractmethod
    def __init__(self, model_name: str, history_file_path: Path):
        pass

    @abstractmethod
    def chat_stream(self, message: str) -> str:
        pass

    @abstractmethod
    def chat(self, message: str) -> str:
        pass

    @abstractmethod
    def _load_history(self):
        pass

    @abstractmethod
    def _save_history(self):
        pass
