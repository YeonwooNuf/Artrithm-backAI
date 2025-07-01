from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from app.loaders.pdf_loader import load_pdf_and_create_vectorstore
from app.loaders.vector_loader import load_vectorstore
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os

router = APIRouter()

class LLMQuery(BaseModel):
    artworkId: int
    question: str

@router.post("/query")
def query_llm(q: LLMQuery):
    pdf_path = f"documents/{q.artworkId}.pdf"
    vector_path = f"vectorstore_cache/{q.artworkId}"

    if not os.path.exists(pdf_path):
        return {"answer": "해당 작품에 대한 설명 파일이 없습니다."}

    if not os.path.exists(vector_path):
        return {"answer": "설명 파일은 있지만 벡터 DB가 없습니다. PDF를 다시 업로드 해주세요."}

    vectorstore = load_vectorstore(q.artworkId)

    # ✅ 벡터 검색
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"score_threshold": 0.7}
    )
    docs = retriever.get_relevant_documents(q.question)

    if not docs:
        return {
            "answer": "해당 질문은 작품과 관련되지 않았습니다.",
            "source_documents": []
        }

    # ✅ context 만들기
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_template = """
    너는 미술관에서 작품을 관람하는 사용자들에게 미술 작품에 대한 설명을 도와주는 AI야.
    아래 문서를 참고해서 사용자의 질문에 대해 **자연스럽고 친절한 한국어(Korean)로** 답변해줘.
    문서 내용:
    {context}

    질문:
    {question}

    답변 (한국어로):
    """
    prompt = PromptTemplate.from_template(prompt_template.strip())

    llm = Ollama(model="gemma:2b")
    formatted_prompt = prompt.format(context=context, question=q.question)
    answer = llm.invoke(formatted_prompt)

    return {
        "answer": answer,
        "source_documents": [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page"),
                "score": doc.metadata.get("score", 0.0)
            }
            for doc in docs
        ]
    }
