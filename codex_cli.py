#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Codex 命令行工具
提供简单的命令行接口来使用 OpenAI 和 Tiktoken 库
"""

import argparse
import os
import sys
import openai
import tiktoken

def count_tokens(text, model="gpt-4"):
    """计算文本的 token 数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        token_count = len(encoding.encode(text))
        return token_count
    except Exception as e:
        print(f"计算 token 数量时出错: {e}")
        return None

def encode_text(text, model="gpt-4"):
    """编码文本为 tokens"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return tokens
    except Exception as e:
        print(f"编码文本时出错: {e}")
        return None

def decode_tokens(tokens, model="gpt-4"):
    """解码 tokens 为文本"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        text = encoding.decode(tokens)
        return text
    except Exception as e:
        print(f"解码 tokens 时出错: {e}")
        return None

def list_models():
    """列出可用的模型"""
    try:
        client = openai.OpenAI()
        models = client.models.list()
        print("可用的 OpenAI 模型:")
        for model in models:
            print(f"- {model.id}")
    except Exception as e:
        print(f"获取模型列表时出错: {e}")
        print("请确保已设置 OPENAI_API_KEY 环境变量")

def main():
    parser = argparse.ArgumentParser(description="OpenAI Codex 命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 版本命令
    parser.add_argument("--version", action="version", version="Codex CLI 1.0.0")
    
    # 计数 tokens 命令
    count_parser = subparsers.add_parser("count", help="计算文本的 token 数量")
    count_parser.add_argument("text", help="要计算的文本")
    count_parser.add_argument("--model", default="gpt-4", help="使用的模型 (默认: gpt-4)")
    
    # 编码命令
    encode_parser = subparsers.add_parser("encode", help="将文本编码为 tokens")
    encode_parser.add_argument("text", help="要编码的文本")
    encode_parser.add_argument("--model", default="gpt-4", help="使用的模型 (默认: gpt-4)")
    
    # 解码命令
    decode_parser = subparsers.add_parser("decode", help="将 tokens 解码为文本")
    decode_parser.add_argument("tokens", nargs="+", type=int, help="要解码的 tokens (空格分隔)")
    decode_parser.add_argument("--model", default="gpt-4", help="使用的模型 (默认: gpt-4)")
    
    # 列出模型命令
    subparsers.add_parser("models", help="列出可用的 OpenAI 模型")
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行命令
    if args.command == "count":
        count = count_tokens(args.text, args.model)
        if count is not None:
            print(f"文本: {args.text}")
            print(f"模型: {args.model}")
            print(f"Token 数量: {count}")
    
    elif args.command == "encode":
        tokens = encode_text(args.text, args.model)
        if tokens is not None:
            print(f"文本: {args.text}")
            print(f"模型: {args.model}")
            print(f"Tokens: {tokens}")
    
    elif args.command == "decode":
        text = decode_tokens(args.tokens, args.model)
        if text is not None:
            print(f"Tokens: {args.tokens}")
            print(f"模型: {args.model}")
            print(f"文本: {text}")
    
    elif args.command == "models":
        list_models()

if __name__ == "__main__":
    main()