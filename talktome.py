import os
import uuid
import logging
from collections import defaultdict

from dotenv import load_dotenv
from flask import Flask, render_template, request, session
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
from src.prompt import system_prompt  # must contain {context}

load_dotenv()

for key in ("PINECONE_API_KEY", "OPENAI_API_KEY", "FLASK_SECRET_KEY"):
    if not os.environ.get(key):
        raise RuntimeError(f"Missing environment variable: {key}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

MAX_TURNS = 6  # number of past exchanges kept per session

# In-memory store; use Redis (or similar) if you run multiple workers.
chat_histories = defaultdict(list)

embeddings = download_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medi-chatbot",
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chat_model = ChatOpenAI(model="gpt-4o", temperature=0, timeout=30, max_retries=2)

# 1) Rewrite follow-up questions into standalone queries for retrieval
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, rewrite the question "
     "so it can be understood without the history. Do NOT answer it. "
     "If it is already standalone, return it unchanged."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, contextualize_prompt
)

# 2) Answer using retrieved context plus the conversation so far
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def get_sid() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = (request.form.get("msg") or "").strip()
    if not user_msg:
        return "Please enter a message.", 400

    history = chat_histories[get_sid()]
    try:
        result = rag_chain.invoke({"input": user_msg, "chat_history": history})
    except Exception:
        logger.exception("RAG chain failed")
        return "Sorry, something went wrong. Please try again.", 500

    answer = str(result["answer"])
    history.extend([HumanMessage(content=user_msg), AIMessage(content=answer)])
    del history[:-MAX_TURNS * 2]  # keep only the most recent turns
    return answer


@app.route("/reset", methods=["POST"])
def reset():
    chat_histories.pop(get_sid(), None)
    return "", 204


if __name__ == "__main__":
    # For local dev only. In production: gunicorn -w 2 -b 0.0.0.0:8080 app:app
    app.run(host="127.0.0.1", port=8080, debug=False)