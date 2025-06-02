from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_vectorstore(artwork_id: int):
    embeddings = OllamaEmbeddings(model="gemma:2b")
    path = f"vectorstore_cache/{artwork_id}"
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # 🔥 중요: 명시적으로 허용
    )
