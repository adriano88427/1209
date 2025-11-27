# OpenAI Codex 命令行工具

这是一个简单的命令行工具，用于与 OpenAI Codex 相关的功能，包括计算文本的 token 数量、编码和解码文本等。

## 安装

1. 确保已安装 Python 3.7 或更高版本
2. 安装必要的依赖：
   ```
   pip install openai tiktoken
   ```

## 使用方法

### 检查版本
```
.\codex.bat --version
```

### 计算文本的 token 数量
```
.\codex.bat count "你的文本"
```
示例：
```
.\codex.bat count "Hello, this is a test of the OpenAI Codex tokenizer."
```

### 将文本编码为 tokens
```
.\codex.bat encode "你的文本"
```
示例：
```
.\codex.bat encode "Hello, world!"
```

### 将 tokens 解码为文本
```
.\codex.bat decode token1 token2 token3 ...
```
示例：
```
.\codex.bat decode 9906 11 420 374 264
```

### 列出可用的 OpenAI 模型
```
.\codex.bat models
```
注意：此功能需要设置 OPENAI_API_KEY 环境变量

### 指定模型
默认使用 gpt-4 模型，您可以通过 --model 参数指定其他模型：
```
.\codex.bat count "你的文本" --model gpt-3.5-turbo
```

## 环境变量

要使用需要 API 密钥的功能（如列出模型），请设置环境变量：
```
set OPENAI_API_KEY=your-api-key-here
```

## 示例

1. 计算中文文本的 token 数量：
   ```
   .\codex.bat count "这是一个测试文本，用于计算 token 数量。"
   ```

2. 编码文本为 tokens：
   ```
   .\codex.bat encode "Hello, world!"
   ```

3. 解码 tokens 为文本：
   ```
   .\codex.bat decode 9906 11 420 374 264
   ```

## 注意事项

- 在 PowerShell 中，需要使用 `.\codex.bat` 而不是 `codex.bat`
- 如果要在任何目录下使用 `codex` 命令，可以将包含 `codex.bat` 的目录添加到系统的 PATH 环境变量中
- 某些功能需要有效的 OpenAI API 密钥

## 故障排除

如果遇到 "无法将 'codex.bat' 项识别为 cmdlet、函数、脚本文件或可运行程序的名称" 错误，请确保：
1. 您在正确的目录中（包含 codex.bat 文件的目录）
2. 使用 `.\codex.bat` 而不是 `codex.bat`（在 PowerShell 中）