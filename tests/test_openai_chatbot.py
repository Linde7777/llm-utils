import unittest
import tempfile
import shutil
from pathlib import Path
import json
from unittest.mock import Mock, patch
from openai_chatbot import OpenAIChatbot
import pytest
import os

@pytest.fixture
def mock_history_file(tmp_path):
    """创建一个临时的历史文件用于测试"""
    history_file = tmp_path / "test_history.json"
    initial_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(initial_history, f)
    return history_file

@pytest.fixture
def chatbot(mock_history_file):
    """创建一个测试用的chatbot实例"""
    return OpenAIChatbot(
        model_name="gpt-3.5-turbo",
        history_file=mock_history_file,
        api_key="test_key"
    )

class TestOpenAIChatbot(unittest.TestCase):
    """OpenAIChatbot的测试类"""
    
    def setUp(self):
        """测试前的设置"""
        # 创建临时目录和文件
        self.temp_dir = tempfile.mkdtemp()
        self.history_file = Path(self.temp_dir) / "test_history.json"
        initial_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(initial_history, f)
            
        self.chatbot = OpenAIChatbot(
            model_name="gpt-3.5-turbo",
            history_file=self.history_file,
            api_key="test_key"
        )

    def tearDown(self):
        """测试后的清理"""
        shutil.rmtree(self.temp_dir)

    def test_init_missing_history_file(self):
        """测试历史文件不存在时的错误处理"""
        with self.assertRaises(FileNotFoundError):
            OpenAIChatbot(
                model_name="gpt-3.5-turbo",
                history_file=Path("nonexistent.json"),
                api_key="test_key"
            )

    def test_chat(self):
        """测试普通聊天功能"""
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="This is a test response"))
        ]
        
        # 替换 chatbot.client.chat.completions.create 方法
        with patch.object(self.chatbot.client.chat.completions, 'create', return_value=mock_response):
            response = self.chatbot.chat("Test message", should_print=False)
            
            self.assertEqual(response, "This is a test response")
            # 验证历史记录是否正确更新
            self.assertEqual(len(self.chatbot.chat_history), 3)  # system prompt + user message + assistant response
            self.assertEqual(self.chatbot.chat_history[-2]["role"], "user")
            self.assertEqual(self.chatbot.chat_history[-2]["content"], "Test message")
            self.assertEqual(self.chatbot.chat_history[-1]["role"], "assistant")
            self.assertEqual(self.chatbot.chat_history[-1]["content"], "This is a test response")

    def test_chat_stream(self):
        """测试流式聊天功能"""
        mock_chunk = Mock()
        mock_chunk.choices = [Mock(delta=Mock(content="Test "))]
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock(delta=Mock(content="response"))]
        
        mock_stream = [mock_chunk, mock_chunk2]
        
        with patch.object(self.chatbot.client.chat.completions, 'create', return_value=mock_stream):
            response = self.chatbot.chat_stream("Test message", should_print=False)
            
            self.assertEqual(response, "Test response")
            # 验证历史记录是否正确更新
            self.assertEqual(len(self.chatbot.chat_history), 3)
            self.assertEqual(self.chatbot.chat_history[-2]["role"], "user")
            self.assertEqual(self.chatbot.chat_history[-2]["content"], "Test message")
            self.assertEqual(self.chatbot.chat_history[-1]["role"], "assistant")
            self.assertEqual(self.chatbot.chat_history[-1]["content"], "Test response")

    def test_error_handling(self):
        """测试错误处理装饰器"""
        with patch.object(self.chatbot.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with self.assertRaises(ConnectionError):
                self.chatbot.chat("Test message")

    def test_history_persistence(self):
        """测试聊天历史的持久化"""
        # 创建一个新的chatbot实例并发送消息
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Test response"))
        ]
        
        with patch.object(self.chatbot.client.chat.completions, 'create', return_value=mock_response):
            self.chatbot.chat("Test message", should_print=False)
        
        # 创建一个新的实例，验证历史记录是否正确加载
        chatbot2 = OpenAIChatbot(
            model_name="gpt-3.5-turbo",
            history_file=self.history_file,
            api_key="test_key"
        )
        
        self.assertEqual(len(chatbot2.chat_history), 3)
        self.assertEqual(chatbot2.chat_history[-2]["content"], "Test message")
        self.assertEqual(chatbot2.chat_history[-1]["content"], "Test response")

    def test_system_prompt_with_empty_history(self):
        """测试空历史记录时是否正确添加系统提示"""
        # 创建一个空的历史文件
        empty_history_file = Path(self.temp_dir) / "empty_history.json"
        with open(empty_history_file, 'w', encoding='utf-8') as f:
            json.dump([], f)  # 写入空数组
            
        custom_prompt = "Custom system prompt"
        chatbot = OpenAIChatbot(
            model_name="gpt-3.5-turbo",
            history_file=empty_history_file,
            system_prompt=custom_prompt,
            api_key="test_key"
        )
        
        # 验证是否正确添加了系统提示
        self.assertEqual(len(chatbot.chat_history), 1)
        self.assertEqual(chatbot.chat_history[0]["role"], "system")
        self.assertEqual(chatbot.chat_history[0]["content"], custom_prompt)

    def test_system_prompt_with_existing_history(self):
        """测试非空历史记录时是否保持原样"""
        # 创建一个包含自定义历史记录的文件
        existing_history_file = Path(self.temp_dir) / "existing_history.json"
        existing_history = [
            {"role": "system", "content": "Original system prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        with open(existing_history_file, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f)
            
        # 使用不同的系统提示创建chatbot
        chatbot = OpenAIChatbot(
            model_name="gpt-3.5-turbo",
            history_file=existing_history_file,
            system_prompt="New system prompt",  # 这个不应该被使用
            api_key="test_key"
        )
        
        # 验证历史记录保持不变
        self.assertEqual(len(chatbot.chat_history), 3)
        self.assertEqual(chatbot.chat_history[0]["content"], "Original system prompt")
        self.assertEqual(chatbot.chat_history[1]["content"], "Hello")
        self.assertEqual(chatbot.chat_history[2]["content"], "Hi there")

if __name__ == '__main__':
    unittest.main() 