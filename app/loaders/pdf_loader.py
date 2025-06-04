from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf_and_create_vectorstore(pdf_path: str, artwork_id: int):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="gemma:2b")

    # ✅ 로컬에 저장
    save_path = f"vectorstore_cache/{artwork_id}"
    FAISS.from_documents(split_docs, embeddings).save_local(save_path)
