from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_documents, split_documents,download_openai_embeddings
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

print(f"Number of text chunks: {len(text_chunks)}")
for i, doc in enumerate(text_chunks[:5]):
    print(f"\nDocument {i + 1}")
    print("Content length:", len(doc.page_content))
    print("Metadata:", doc.metadata)


embeddings = download_openai_embeddings()
# vector=embeddings.embed_query("Hello world")
# print(len(vector))

pincecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pincecone_api_key)

from pinecone import ServerlessSpec


index_name="openai-medi-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)
index.delete(delete_all=True)

# ----------------------------------
# 5. Connect to Pinecone
# ----------------------------------

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

# ----------------------------------
# 6. Upload in controlled batches
# ----------------------------------
BATCH_SIZE =10
for i in range(0, len(text_chunks), BATCH_SIZE):

    batch = text_chunks[i:i + BATCH_SIZE]

    vectorstore.add_documents(
        documents=batch
    )

    uploaded = min(i + BATCH_SIZE, len(text_chunks))

    print(f"Uploaded {uploaded}/{len(text_chunks)} chunks")

print("PDF ingestion completed successfully!")


# docsearch = PineconeVectorStore.from_documents(
#     documents=text_chunks,
#     index_name=index_name,
#     embedding=embeddings,
#     batch_size=10,
# ) 