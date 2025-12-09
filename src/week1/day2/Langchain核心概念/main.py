import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call, dynamic_prompt
# from langchain_community.chat_models import ChatZhipuAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("ZAI_API_KEY")
api_base = os.getenv("ZAI_API_BASE")

class Context(TypedDict):
    user_role:str

zai_model = ChatOpenAI(
    model="glm-4.6",
    max_tokens=1000,
    api_key=api_key,
    base_url=api_base,
)
ollama_model = ChatOllama(
    model="qwen3:0.6b",
    # base_url="http",
    max_tokens=1000,
)

@wrap_model_call
def dynamic_model(request: ModelRequest, handler) -> ModelResponse:
    """ 选择模型"""
    message_count = len(request.state["messages"])

    if message_count < 10:
        # Use an advanced model for longer conversations
        model = ollama_model
    else:
        model = zai_model
    return handler(request.override(model=model))

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input andtry again.({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.get("user_role","user")
    base_prompt = "你是考古专家"

    if user_role == "考古":
        return base_prompt
    elif user_role == "养生":
        return "你是健康饮食专家"

    return base_prompt

@tool
def search(query:str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(
    zai_model,
    tools=[search, get_weather],
    middleware=[dynamic_model, handle_tool_errors,user_role_prompt],
    context_schema=Context,
    # system_prompt="你是极品猫娘"
)

res = agent.invoke({
    "messages": [{"role":"user","content":"介绍你自己"}]},
    context={"user_role":"养生"}
)

print(res)
