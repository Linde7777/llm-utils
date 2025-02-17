import pytest
from pathlib import Path
import json
from openai_chatbot import OpenAIChatbot
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_openai():
    with patch('openai_chatbot.OpenAI') as mock:
        yield mock

@pytest.fixture
def temp_history_file(tmp_path):
    return tmp_path / "test_history.json"

def test_init(mock_openai, temp_history_file):
    chatbot = OpenAIChatbot("gpt-3.5-turbo", temp_history_file, "test-key")
    assert chatbot.model_name == "gpt-3.5-turbo"
    assert chatbot.api_key == "test-key"
    mock_openai.assert_called_once_with(api_key="test-key", base_url=None)

def test_chat(mock_openai, temp_history_file):
    # Mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Test response"
    mock_openai.return_value.chat.completions.create.return_value = mock_response
    
    chatbot = OpenAIChatbot("gpt-3.5-turbo", temp_history_file, "test-key")
    response = chatbot.chat("Test message", should_print=False)
    
    assert response == "Test response"
    assert len(chatbot.chat_history) == 2
    assert chatbot.chat_history[0]["content"] == "Test message"
    assert chatbot.chat_history[1]["content"] == "Test response"

def test_load_history_file_not_found(temp_history_file):
    with pytest.raises(FileNotFoundError):
        chatbot = OpenAIChatbot("gpt-3.5-turbo", temp_history_file, "test-key")

def test_load_save_history(temp_history_file):
    # Create empty history file first
    temp_history_file.write_text("[]", encoding='utf-8')
    
    # Test saving
    chatbot = OpenAIChatbot("gpt-3.5-turbo", temp_history_file, "test-key")
    chatbot.chat_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    chatbot._save_history()
    
    # Test loading
    new_chatbot = OpenAIChatbot("gpt-3.5-turbo", temp_history_file, "test-key")
    assert len(new_chatbot.chat_history) == 2
    assert new_chatbot.chat_history[0]["content"] == "Hello"
    assert new_chatbot.chat_history[1]["content"] == "Hi"

def test_load_invalid_json(temp_history_file):
    # Create file with invalid JSON
    temp_history_file.write_text("invalid json", encoding='utf-8')
    
    with pytest.raises(json.JSONDecodeError):
        chatbot = OpenAIChatbot("gpt-3.5-turbo", temp_history_file, "test-key")

def test_error_handling(mock_openai, temp_history_file):
    mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
    
    chatbot = OpenAIChatbot("gpt-3.5-turbo", temp_history_file, "test-key")
    with pytest.raises(ConnectionError):
        chatbot.chat("Test message") 