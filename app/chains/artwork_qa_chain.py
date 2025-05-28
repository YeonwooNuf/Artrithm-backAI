# 질문이 들어왔을 때,
# 1) 관련 문단을 벡터 DB에서 검색하고
# 2) LLM(Ollama llama3)을 이용해 답변을 생성하는 RAG 체인을 생성

from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# 주어진 벡터 저장소를 기반으로 Retrieval QA 체인을 생성하는 함수
def get_qa_chain(vectorstore):
    # 1. 벡터 저장소에서 유사 문서를 검색할 수 있도록 검색기 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"score_threshold": 0.5}  # 유사도 기준 설정
    )

    # 2. 로컬 Ollama LLM 모델 호출 (사전에 ollama run mistral 실행 필요)
    llm = Ollama(model="gemma:2b")

    # 3. 검색기 + LLM 조합으로 QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
