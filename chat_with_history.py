from flask import Flask, render_template, jsonify, request
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# from src.prompt import *
from src.helper import download_openai_embeddings
import os
app = Flask(__name__)
# Assume `llm` and `retriever` (e.g., from Pinecone or Chroma) are already initialized
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_openai_embeddings()


NUM_TURNS = 3

trimmer = trim_messages(
    max_tokens=NUM_TURNS * 2,  # 1 turn = 2 messages (Human + AI)
    strategy="last",           # Keep the most recent messages
    token_counter=len,         # Count individual messages instead of tokens
    include_system=True,       # Preserve system prompt if present
    start_on="human",          # Guarantee the window starts with a HumanMessage
    # Optional: ensures history ends cleanly before the new user input
    end_on=("human", "tool"),
)


index_name = "openai-medi-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o")

# 1. Prompt to rewrite follow-up questions into standalone queries
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create the history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 2. Prompt to answer the question using retrieved context + chat history
qa_system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the answer concise.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Combine retrieval and generation
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)

# 3. Manage session-based chat history
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


trimmed_rag_chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: trimmer.invoke(x["chat_history"])
    )
    | rag_chain
)

conversational_rag_chain = RunnableWithMessageHistory(
    trimmed_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# 4. Run the conversation
config = {"configurable": {"session_id": "user_session_1"}}

# res1 = conversational_rag_chain.invoke({"input": "What is Acne"}, config=config)
# print(res1["answer"])

# # Follow-up question relying on previous turn
# res2 = conversational_rag_chain.invoke({"input": "How is it treated?"}, config=config)
# print(res2["answer"])


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    # print(chat.invoke({"input": "My name is Tomfo."}, config=cfg).content)
    response = conversational_rag_chain.invoke({"input": msg}, config=config)
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
