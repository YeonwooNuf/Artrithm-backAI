from fastapi import APIRouter, Query
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ 이거로 교체

router = APIRouter()

# ✅ 1. FAISS 인덱스 로딩
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vectorstore = FAISS.load_local(
    "vectorstore_cache/exhibitions",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ✅ 2. 추천 API
@router.get("/recommend")
async def recommend(query: str = Query(...)):
    results = retriever.get_relevant_documents(query)

    return {
        "recommended": [
            {
                "id": doc.metadata.get("id"),
                "content": doc.page_content,
                "rank": i + 1
            }
            for i, doc in enumerate(results)
        ]
    }
