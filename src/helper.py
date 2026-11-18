from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings


# load pdf files from a directory and return a list of documents
def load_pdf_files(data):
    loader = DirectoryLoader(
    data, 
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_documents(docs: List[Document]) -> List[Document]:
    minimal_documents: List[Document] = []
    for doc in docs:
        src = doc.metadata.get('source', '')
        minimal_documents.append(
            Document(
                page_content=doc.page_content, 
                metadata={'source': src}))
    return minimal_documents

# split the documents into chunks
def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap        
    )
    return text_splitter.split_documents(documents)


def download_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name)
    
    return embeddings
embedding = download_embeddings()