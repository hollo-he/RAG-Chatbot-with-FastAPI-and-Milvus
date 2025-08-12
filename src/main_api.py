import asyncio
rag_chain = None
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from rag_chain_provider import get_rag_chain_with_memory
from langchain_core.messages import HumanMessage,AIMessage

app = FastAPI(
    title="RAG 问答机器人API",
    description="一个可以通过网络调用的、基于私有知识库的问答服务。",
    version="1.0.0",
)

class ChatRequest(BaseModel):
    question: str
    chat_history: list[tuple[str,str]] = []

class ChatResponse(BaseModel):
    answer: str

print("正在加载 RAG chain，请稍候...")
rag_chain = get_rag_chain_with_memory()
print("RAG chain 已加载完成，API 服务准备就绪！")

@app.post("/chat",response_model=ChatResponse)
def chat_with_memory(request:ChatRequest):
    converted_chat_history = []
    for user_msg,ai_msg in request.chat_history:
        converted_chat_history.append(HumanMessage(content=user_msg))
        converted_chat_history.append(AIMessage(content=ai_msg))
    response = rag_chain.invoke({
        "input":request.question,
        "chat_history":converted_chat_history
    })
    return ChatResponse(answer=response['answer'])
# --- e. (可选) 添加一个根路径用于测试服务是否启动 ---
@app.get("/")
def read_root():
    return {"message": "欢迎使用 RAG 问答机器人 API，请访问 /docs 查看详情。"}