from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_documents, split_documents, download_huggingface_embeddings, download_openai_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_files(data='./knowledge_base')
filtered_data = filter_to_minimal_documents(extracted_data)
text_chunks = split_documents(filtered_data)

embeddings = download_huggingface_embeddings()

pincecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pincecone_api_key)

from pinecone import ServerlessSpec
index_name="medi-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)