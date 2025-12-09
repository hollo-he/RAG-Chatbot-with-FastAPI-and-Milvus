import chromadb
from chromadb import Settings
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings


Settings.llm = Ollama(
    model="qwen3:0.6b",
    base_url="http://localhost:11434",
)

# 1. 读取文档
documents = SimpleDirectoryReader(
    input_files=["../document/learning-roadmap-development.md"]
).load_data()

# 2. 使用 HuggingFace embedding（最稳定）
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5"
)

# 3. Chroma 向量库
db = chromadb.PersistentClient(path="../document/chroma_db")
collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. 写入向量
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

# 5. 重新加载向量库（测试）
db2 = chromadb.PersistentClient(path="../document/chroma_db")
collection2 = db2.get_or_create_collection("quickstart")
vector_store2 = ChromaVectorStore(chroma_collection=collection2)

index2 = VectorStoreIndex.from_vector_store(
    vector_store2,
    embed_model=embed_model,
)

# 6. 查询
query_engine = index2.as_query_engine()
response = query_engine.query("我第 2 周的学习内容是什么")
print(response)
