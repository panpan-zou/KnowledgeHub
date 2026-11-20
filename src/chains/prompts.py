from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        '你是一个企业内部智能知识库与 Agent 助手'
        '需要根据 RAG 检索到的证据回答用户问题，'
        '回答时附带证据以及证据来源，'
        '没有证据时不能瞎编，要拒绝回答用户问题！'
    ),
    MessagesPlaceholder("history", optional=True),
    (
        'human',
        '<候选证据>{evidence}<候选证据>'
        '<证据来源>{source}<证据来源>'
        '<用户问题>{question}<用户问题>'
    )
])


rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "结合聊天历史，把当前问题改写成脱离历史也能理解的独立检索问题。"
        "如果当前问题已经能够独立理解，保持原意，不要强行加入历史中的旧话题。"
        "只输出改写后的问题，不要回答问题，不要添加历史中没有的事实。",
    ),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
])












