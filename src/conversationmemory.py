"""
Conversation buffer memory in LangChain: two approaches.

pip install langchain langchain-openai langchain-core
(for legacy classes on LangChain 1.x also: pip install langchain-classic)

export OPENAI_API_KEY=...   # or swap ChatOpenAI for ChatAnthropic, etc.
"""

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
#  Recommended: RunnableWithMessageHistory + InMemoryChatMessageHistory
#   Same buffer behaviour, but per-session and works with LCEL.
# ---------------------------------------------------------------------------
def buffer_memory():
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm

    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chat = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "user-1"}}
    print(chat.invoke({"input": "My name is Tomfo."}, config=cfg).content)
    print(chat.invoke({"input": "What's my name?"}, config=cfg).content)

    # A different session has its own, empty history
    other = {"configurable": {"session_id": "user-2"}}
    print(chat.invoke({"input": "What's my name?"}, config=other).content)

    print(store["user-1"].messages)


