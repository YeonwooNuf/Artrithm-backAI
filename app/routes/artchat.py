from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from app.loaders.pdf_loader import load_pdf_and_create_vectorstore
from app.loaders.vector_loader import load_vectorstore  # ✅ 벡터스토어 로더 추가
from app.chains.artwork_qa_chain import get_qa_chain
import os

router = APIRouter()

class LLMQuery(BaseModel):
    artworkId: int
    question: str

# ✅ 질문 처리: 벡터를 실시간 생성 ❌, 저장된 것 로드 ⭕
@router.post("/query")
def query_llm(q: LLMQuery):
    pdf_path = f"documents/{q.artworkId}.pdf"
    vector_path = f"vectorstore_cache/{q.artworkId}"

    if not os.path.exists(pdf_path):
        return {"answer": "해당 작품에 대한 설명 파일이 없습니다."}

    if not os.path.exists(vector_path):
        return {"answer": "설명 파일은 있지만 벡터 DB가 없습니다. PDF를 다시 업로드 해주세요."}

    vectorstore = load_vectorstore(q.artworkId)
    qa_chain = get_qa_chain(vectorstore)
    result = qa_chain.invoke(q.question)

    if not result:
        return {"answer": "답변 생성에 실패했습니다."}

    return {"answer": result["result"]}

# ✅ PDF 업로드 및 벡터 DB 생성
@router.post("/upload")
def upload_pdf(artworkId: int = Form(...), file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "PDF 파일만 업로드할 수 있습니다."}

    os.makedirs("documents", exist_ok=True)
    os.makedirs("vectorstore_cache", exist_ok=True)

    pdf_path = f"documents/{artworkId}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    # ✅ 업로드할 때 벡터 저장소 생성 및 디스크에 저장
    load_pdf_and_create_vectorstore(pdf_path, artworkId)

    return {"message": "PDF 업로드 및 벡터 저장 완료"}
