from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional


class BaseChatbot(ABC):
    """Base class for chatbot implementations.
    
    Attributes:
        model_name (str): The name of the model being used
        history_file (Path): Path to the file storing chat history
    """
    
    @abstractmethod
    def __init__(self, model_name: str, history_file: Path, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        """Initialize the chatbot.

        Args:
            model_name: The name of the model to use
            history_file: Path to the file storing chat history
            api_key: Optional API key. If not provided, will try to read from OPENAI_API_KEY environment variable
            base_url: Optional base URL for the API endpoint

        Raises:
            ValueError: If API key is not provided
            FileNotFoundError: If the history file doesn't exist
        """
        if not history_file.exists():
            raise FileNotFoundError(f"History file not found: {history_file}")
            
        self.model_name = model_name
        self.history_file = history_file
        self.chat_history = [Dict[str, str]]
        self._load_history()


    @abstractmethod
    def chat_stream(self, message: str, should_print: bool = True) -> str:
        """Stream the chat response.
        
        Args:
            message: The user's input message
            should_print: Whether to print the response while streaming.
                        Sometimes the print will disturb debugging.
            
        Returns:
            The complete response as a string
            
        Raises:
            ConnectionError: If there's an issue with the model connection
        """
        pass

    @abstractmethod
    def chat(self, message: str, should_print: bool = True) -> str:
        """Send a message and get a response.
        
        Args:
            message: The user's input message
            should_print: Whether to print the response.
                        Sometimes the print will disturb debugging.
        Returns:
            The chatbot's response
        """
        pass

    @abstractmethod
    def _load_history(self) -> None:
        """Load chat history from self.history_file."""
        pass

    @abstractmethod
    def _save_history(self) -> None:
        """Save chat history to self.history_file."""
        pass
