from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vectorstore(artwork_id: int):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    path = f"vectorstore_cache/{artwork_id}"
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
