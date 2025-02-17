import os

from base_chatbot import BaseChatbot
from pathlib import Path
from typing import Optional, Dict
import json
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletion


def handle_openai_errors(func):
    """Decorator to handle OpenAI API errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise ConnectionError(f"Error communicating with OpenAI: {str(e)}")
    return wrapper

class OpenAIChatbot(BaseChatbot):
    def __init__(self, model_name: str, history_file: Path, 
                system_prompt: str = "You are a helpful assistant.",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:
        """Initialize OpenAI chatbot."""
        if not history_file.exists():
            raise FileNotFoundError(f"History file not found: {history_file}")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("API key must be provided either through parameter or OPENAI_API_KEY environment variable")

        self.model_name = model_name
        self.history_file = history_file
        self.chat_history = [{'role': 'system', 'content': system_prompt}]
        self._load_history()
        self.client = OpenAI(api_key=self.api_key, 
            base_url=base_url if base_url else None)

    @handle_openai_errors
    def chat_stream(self, message: str, should_print: bool = True) -> str:
        """Stream chat response from OpenAI."""
        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})
        
        # Create stream response
        stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in self.chat_history],
            stream=True
        )
        
        # Collect response
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                if should_print:
                    print(content, end="", flush=True)
        
        if should_print:
            print()  # New line after response
            
        # Add assistant response to history
        self.chat_history.append({"role": "assistant", "content": full_response})
        self._save_history()
        
        return full_response

    @handle_openai_errors
    def chat(self, message: str, should_print: bool = True) -> str:
        """Send message and get response without streaming."""
        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})
        
        # Get response
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in self.chat_history],
            stream=False
        )
        
        response_text = response.choices[0].message.content or ""
        
        if should_print:
            print(response_text)
            
        # Add assistant response to history
        self.chat_history.append({"role": "assistant", "content": response_text})
        self._save_history()
        
        return response_text

    def _load_history(self) -> None:
        """Load chat history from JSON file.
        """
        with open(self.history_file, 'r', encoding='utf-8') as f:
            self.chat_history = json.load(f)
    

    def _save_history(self) -> None:
        """Save chat history to JSON file."""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
