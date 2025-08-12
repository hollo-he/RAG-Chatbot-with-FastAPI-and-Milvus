import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus
# 我们需要一个底层的 Milvus 客户端来检查集合是否存在
from pymilvus import utility, connections
# --- 使用 ZhipuAI Embedding 模型 ---
# 1. 导入智谱 AI 的 Embedding 类
from langchain_community.embeddings import ZhipuAIEmbeddings

# 2. 从环境变量中获取你的智谱 API Key
#    确保你的 .env 文件里有 ZHIPUAI_API_KEY="sk-xxxx" 这一行
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
if not ZHIPUAI_API_KEY:
    print("错误：请先在 .env 文件中设置 ZHIPUAI_API_KEY。")
    exit()


load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("错误：请先在 .env 文件中设置 GOOGLE_API_KEY。")
    exit()

SOURCE_FILE = "../data/simple.txt" # 确保使用的是包含诗句的 test.txt
COLLECTION_NAME = "my_rag_collection_manual" # 使用一个新的集合名，以作区分
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"

# --- 2. 数据准备（和之前一样） ---
print("--- 1. 加载和分割文档 ---")
loader = TextLoader(SOURCE_FILE, encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=24, chunk_overlap=1)
chunks = text_splitter.split_documents(docs)
print(f"文档已分割成 {len(chunks)} 个文本块。")

# --- 3. 初始化 Embedding 模型（和之前一样） ---
print("正在初始化 ZhipuAI Embedding 模型...")
# 3. 初始化 ZhipuAIEmbeddings
#    它会自动使用你的 API Key
embeddings = ZhipuAIEmbeddings(
    api_key=ZHIPUAI_API_KEY
    # 智谱的 Embedding 模型通常是固定的，不需要指定 model 参数
)
print("ZhipuAI Embedding 模型加载完成。")
# --- 替换完成 ---

# --- 4. 【核心区别】手动控制数据入库流程 ---
print(f"\n--- 3. 手动向量化并逐条存入 Milvus 集合 '{COLLECTION_NAME}' ---")

# a. 首先，连接到底层的 Milvus 服务
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# b. 检查集合是否已存在，如果存在，先删除（保证每次都是干净的入库）
if utility.has_collection(COLLECTION_NAME):
    print(f"发现已存在的集合 '{COLLECTION_NAME}'，正在删除...")
    utility.drop_collection(COLLECTION_NAME)
    print("旧集合已删除。")

# c. 初始化一个空的 Milvus 向量存储对象
#    这次我们不传入任何 documents，只是告诉它配置信息
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
    collection_name=COLLECTION_NAME,
)

# d. 遍历每一个文本块，手动地、一个一个地添加
for i, chunk in enumerate(chunks):
    # .add_documents() 方法接收一个文档列表
    # 我们每次只给它一个文档，强迫它进行单次处理
    vector_store.add_documents([chunk])
    # 打印进度，方便我们观察
    print(f"已存入第 {i+1}/{len(chunks)} 个文本块: '{chunk.page_content[:20]}...'")

print("\n--- 数据手动入库流程完成！---")

# e. 断开底层连接
connections.disconnect("default")