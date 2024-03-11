from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def write(query: str):
    """按照用户要求生成文章"""
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("你是专业的文档写手。你根据客户的要求，写一份文档。输出中文。"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )
    chain = {"query": RunnablePassthrough()} | template | ChatOpenAI() | StrOutputParser()

    return chain.invoke(query)
