# RAG 问答机器人 API (基于 FastAPI, LangChain 和 Milvus)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个功能强大的、支持多轮对话的、基于私有知识库的 RAG (检索增强生成) 问答机器人 API 服务。本项目是 XXX 学习过程中的最终融合项目，展示了如何将 LangChain 的 AI 逻辑与 FastAPI 的后端服务能力完美结合。

## ✨ 项目亮点

*   **先进的 RAG 架构**: 基于最新的 LangChain 表达式语言 (LCEL)，构建了包含“问题改写 -> 检索 -> 生成”的完整 RAG 链。
*   **支持多轮对话**: 实现了对话记忆功能，能够理解上下文，进行连贯的、有意义的多轮对话。
*   **高性能向量数据库**: 使用 Milvus 作为向量数据库，并配置了数据持久化，确保知识库的稳定和高效检索。
*   **生产级 API 服务**: 使用 FastAPI 将 RAG 应用封装成一个健壮的、异步的、符合 REST 风格的 API 服务，并提供交互式 API 文档 (Swagger UI)。
*   **模块化设计**: 项目结构清晰，将 AI 逻辑、API 服务、数据入库等模块解耦，易于维护和扩展。

## 🛠️ 使用的技术栈

*   **后端框架**: FastAPI
*   **AI 框架**: LangChain (LCEL)
*   **大语言模型 (LLM)**: 智谱 AI (GLM-4)
*   **Embedding 模型**: 智谱 AI Embedding
*   **向量数据库**: Milvus
*   **API 服务器**: Uvicorn

## 🚀 如何安装和运行

### 1. 先决条件

*   Python 3.10+
*   Docker 和 Docker Compose (用于运行 Milvus)
*   一个智谱 AI 的 API Key

### 2. 克隆与安装

```bash
# 1. 克隆本项目到本地
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名

# 2. 创建并激活 Python 虚拟环境 (推荐)
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 3. 安装所有 Python 依赖
pip install -r requirements.txt
```

### 3. 配置

```bash
# 1. 复制 .env.example (如果提供) 或手动创建 .env 文件
# cp .env.example .env

# 2. 在 .env 文件中填入你的 API Key
ZHIPUAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 4. 启动依赖服务

```bash
# 启动 Milvus 向量数据库
docker-compose up -d
```

### 5. 数据入库

**重要**: 在第一次运行服务前，你需要先将你的知识库文件处理并存入 Milvus。

```bash
# 1. 将你的知识库源文件 (如 .txt, .pdf) 放入 /data 文件夹。
#    (默认使用 data/sample.txt)

# 2. 运行数据入库脚本
python src/ingest_data.py
```

### 6. 启动 API 服务

```bash
# 运行 FastAPI 应用
uvicorn src.main_api:app --host 0.0.0.0 --port 8080
```

### 7. 访问服务

*   **交互式 API 文档 (Swagger UI)**: [http://localhost:8080/docs](http://localhost:8080/docs)
*   **API 接口**: `POST http://localhost:8080/chat`

---

**[最后一步] 将你的本地项目推送到 GitHub**

1.  打开一个终端，`cd` 到你的项目根目录。
2.  执行以下 Git 命令：
    ```bash
    git init
    git add .
    git commit -m "Initial commit: Project setup and implementation of RAG chatbot API"
    git branch -M main
    git remote add origin https://github.com/你的用户名/你的仓库名.git
    git push -u origin main
    ```