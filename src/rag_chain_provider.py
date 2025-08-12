import os
from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

def get_rag_chain():
    load_dotenv()
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
    if not ZHIPUAI_API_KEY:
        raise ValueError("错误：未在.evn文件找到ZHIPUAI_API_KEY")
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("错误：请先在 .env 文件中设置 GOOGLE_API_KEY。")
        exit()
    COLLECTION_NAME = "my_rag_collection_manual"
    MILVUS_HOST = "127.0.0.1"
    MILVUS_PORT = "19530"

    embeddings = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY)

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"host":MILVUS_HOST,"post":MILVUS_PORT},
        collection_name=COLLECTION_NAME
    )
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


    
    template = """
    你是一个专业的问答助手。请根据下面提供的“背景资料”，用简洁、清晰的语言来回答“问题”。
    如果背景资料中没有足够的信息来回答问题，请直接说“根据我所掌握的资料，无法回答该问题。”，不要编造答案。

    背景资料:
    {context}

    问题:
    {question}

    回答:
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context":retriever,"question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

    """
    这是一个新的“工厂函数”，专门构建并返回一个配置完整的、
    带有对话记忆功能的 RAG 链。
    """
def get_rag_chain_with_memory():
    load_dotenv()
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
    if not ZHIPUAI_API_KEY:
        raise ValueError("错误：未在.evn文件找到ZHIPUAI_API_KEY")
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("错误：请先在 .env 文件中设置 GOOGLE_API_KEY。")
        exit()
    COLLECTION_NAME = "my_rag_collection_manual"
    MILVUS_HOST = "127.0.0.1"
    MILVUS_PORT = "19530"

    embeddings = ZhipuAIEmbeddings(api_key=ZHIPUAI_API_KEY)

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"host":MILVUS_HOST,"post":MILVUS_PORT},
        collection_name=COLLECTION_NAME
    )
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """参考下面的对话历史和用户提出的最新问题，这个问题可能会引用对话历史中的上下文。
            请将这个问题改写成一个独立的、完整的、无需参考对话历史就能理解的新问题。
            注意：你只需要改写问题，不要回答问题。如果问题本身已经很完整，无需改写，就直接返回原问题。"""),

            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """你是一个专业的问答助手。
            请使用下面提供的“背景资料”来回答用户提出的“问题”。
            如果找不到答案，请直接说“根据我所掌握的资料，无法回答您的问题。”

            背景资料:
            {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,retriever,contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

    return rag_chain


# 为了方便测试，我们可以加一个主入口
# ...existing code...
if __name__ == '__main__':
    # 这个测试能确保我们的 chain 工厂函数是正常工作的
    try:
        chain = get_rag_chain_with_memory()
        print("RAG chain 创建成功！")
        # 正确的输入格式
        response = chain.invoke({
            "input": "文档里有提到白日依山尽吗？",
            "chat_history": []
        })
        print("测试调用成功，回答：", response)
    except Exception as e:
        print("测试时出错:", e)
# ...existing code...