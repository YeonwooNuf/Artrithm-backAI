from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

def get_qa_response(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"score_threshold": 0.7}
    )

    docs = retriever.get_relevant_documents(question)

    if not docs or len(docs) == 0:
        return {
            "answer": "해당 질문은 작품과 관련되지 않았습니다.",
            "source_documents": []
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_template = """
    너는 미술관에서 작품을 관람하는 사용자들에게 미술 작품에 대한 설명을 도와주는 AI야.
    아래 문서를 참고해서 사용자의 질문에 대해 **자연스럽고 친절한 한국어(Korean)로** 답변해줘.
    **하지만 문서 내용의 작품과 화가의 정보와 관련 없는 질문이라면 반드시 다음처럼 답변해야 해
    : "해당 질문은 작품과 관련되지 않았습니다."**

    문서 내용:
    {context}

    질문:
    {question}

    답변 (한국어로):
    """

    prompt = PromptTemplate.from_template(prompt_template.strip())
    llm = Ollama(model="gemma:2b")

    formatted_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(formatted_prompt)

    return {
        "answer": answer,
        "source_documents": docs
    }
