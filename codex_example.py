#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAI Codex 示例脚本
演示如何使用 OpenAI 和 Tiktoken 库
"""

import openai
import tiktoken

def test_tiktoken():
    """测试 Tiktoken 分词器"""
    print("=== Tiktoken 分词器测试 ===")
    
    # 创建编码器
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 使用的编码器
    
    # 示例文本
    text = "Hello, this is a test of the OpenAI Codex tokenizer."
    
    # 编码文本
    encoded = encoding.encode(text)
    print(f"原始文本: {text}")
    print(f"编码后的 token 数量: {len(encoded)}")
    print(f"编码结果: {encoded}")
    
    # 解码文本
    decoded = encoding.decode(encoded)
    print(f"解码后的文本: {decoded}")
    print()

def test_openai_client():
    """测试 OpenAI 客户端（需要 API 密钥）"""
    print("=== OpenAI 客户端测试 ===")
    
    try:
        # 这里需要设置您的 OpenAI API 密钥
        # client = openai.OpenAI(api_key="your-api-key-here")
        print("OpenAI 客户端已安装，但需要 API 密钥才能使用")
        print("要使用 OpenAI API，请:")
        print("1. 在 https://platform.openai.com/ 获取 API 密钥")
        print("2. 设置环境变量 OPENAI_API_KEY 或在代码中提供密钥")
        print("3. 参考 OpenAI 文档: https://platform.openai.com/docs/api-reference")
    except Exception as e:
        print(f"初始化 OpenAI 客户端时出错: {e}")
    print()

def count_tokens_from_text(text, model="gpt-5.1-codex"):
    """计算文本的 token 数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        token_count = len(encoding.encode(text))
        return token_count
    except Exception as e:
        print(f"计算 token 数量时出错: {e}")
        return None

if __name__ == "__main__":
    print("OpenAI Codex 相关库安装验证脚本\n")
    
    # 测试 Tiktoken
    test_tiktoken()
    
    # 测试 OpenAI 客户端
    test_openai_client()
    
    # 计算示例文本的 token 数量
    sample_text = "这是一个测试文本，用于计算 token 数量。"
    tokens = count_tokens_from_text(sample_text)
    if tokens is not None:
        print(f"示例文本 '{sample_text}' 的 token 数量: {tokens}")
    
    print("\n脚本执行完成！")
