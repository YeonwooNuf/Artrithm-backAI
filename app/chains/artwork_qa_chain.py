from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def get_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"score_threshold": 0.7}
    )

    llm = Ollama(model="gemma:2b")

    # ✅ 한국어 응답 유도 프롬프트 템플릿
    prompt_template = """
    너는 미술 작품에 대한 설명을 도와주는 AI야.
    아래 문서를 참고해서 사용자의 질문에 대해 **자연스럽고 친절한 한국어로** 답변해줘.
    미술 작품에 관련되지 않은 질문에는 제공되지 않는 내용이라고 답변해줘.

    문서 내용:
    {context}

    질문:
    {question}

    답변 (한국어로):
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.strip()
    )

    # ✅ 프롬프트를 포함한 QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain
