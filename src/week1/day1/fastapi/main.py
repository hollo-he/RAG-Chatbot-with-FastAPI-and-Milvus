from fastapi import FastAPI
from src.week1.day2.Langchain核心概念.main import *

app = FastAPI()

@app.get("/query")
async def read_root(query: str):
    res = agent.invoke({
    "messages": [{"role":"user","content":query}]},
    context={"user_role":"养生"})
    return res