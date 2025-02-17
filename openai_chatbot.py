import os

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

class OpenAIChatbot:
    def __init__(self, model_name: str, history_file: Path, 
                system_prompt: str = "You are a helpful assistant.",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:

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
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        debug时如果开启了print，当你把terminal内容发送给cursor后，
        可能会让它对bug源头产生误解（cursor可能没注意到这个函数开启了print）
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        self.chat_history.append({"role": "user", "content": message})
        
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat_history,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                if should_print:
                    print(content, end="", flush=True)
        
        if should_print:
            print()
            
        self.chat_history.append({"role": "assistant", "content": full_response})
        self._save_history()
        
        return full_response

    @handle_openai_errors
    def chat(self, message: str, should_print: bool = True) -> str:
        """Without streaming. 

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        debug时如果开启了print，当你把terminal内容发送给cursor后，
        可能会让它对bug源头产生误解（cursor可能没注意到这个函数开启了print）
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        """
        self.chat_history.append({"role": "user", "content": message})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat_history,
            stream=False
        )
        
        response_text = response.choices[0].message.content or ""
        
        if should_print:
            print(response_text)
            
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
