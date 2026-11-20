from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from src.chains.prompts import rag_prompt, chat_prompt
from src.retrieval.retriever import get_retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")
base_url = os.getenv("BASE_URL")

# 通用模型
model = init_chat_model(
    model=model_name,
    api_key=api_key,
    base_url=base_url,
    model_provider='openai'
)

# 获取向量库查询
retriever = get_retriever()
# 获取存放历史记忆的容器
history = InMemoryChatMessageHistory()

# 将问题去空格
def clean_question(question):
    question = question.strip()
    if not question:
        raise ValueError('输入不能为空')
    return question
clean_question = RunnableLambda(clean_question)

# 定义进入 rag 提示词模板的字典
def rag(question):
    return {
        'question': question,
        'history': history.messages,
    }
rag = RunnableLambda(rag)

# 定义进入 chat 提示词模板的字典
def chat(data:dict):
    return {
        'question': data['question'],
        'evidence': data['evidence'],
        'source': data['source'],
        'history': history.messages,
    }
chat = RunnableLambda(chat)

# 从 retriever 的列表结果中，整合我们所需要的信息
def get_evidence(datas):
    evidence = []
    source = []
    for index, data in enumerate(datas, 1):
        evidence.append(f'[证据{index}] {data.page_content}')
        source.append(data.metadata['source'])
    return {
        'evidence': '\n'.join(evidence),
        'source': source
    }
get_evidence = RunnableLambda(get_evidence)

# 模型结合上下文，优化进入 retriever 查询的句子
def get_content(message):
    print(message.content)
    return message.content
get_content = RunnableLambda(get_content)


# 得到一个含有原始问题，证据文本，证据来源的字典
rag_chain = clean_question | RunnableParallel(
    question=RunnablePassthrough(),
    evidence=rag | rag_prompt | model | get_content | retriever | get_evidence
) | RunnableLambda(
    lambda data: {'question':data['question'], **data['evidence']}
)

# 得到模型最后的结果
query_chain = rag_chain | chat | chat_prompt | model





if __name__ == '__main__':
    user1 = '住宿费可以报销吗？'
    res1 = query_chain.invoke(user1)
    print(res1)

    history.add_messages([
        HumanMessage(user1),
        res1
    ])


    user2 = '能报销多少？'
    res2 = query_chain.invoke(user2)
    print(res2)

    history.add_messages([
        HumanMessage(user2),
        res2
    ])

    print(history.messages)




