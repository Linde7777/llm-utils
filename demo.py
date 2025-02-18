from openai_chatbot import OpenAIChatbot
import os
from pathlib import Path

# Clash for Windows 的端口是7890
# 记得开全局代理
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

if __name__ == "__main__":
    chatbot = OpenAIChatbot(
            model_name="gemini-2.0-flash-lite-preview-02-05",
            history_file=Path("chat_history.json"),
            system_prompt="You are skilled in translating English to Chinese.",
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    chatbot.chat_stream("Hello, how are you?")
    # 此时应当返回中文，而不是英文
