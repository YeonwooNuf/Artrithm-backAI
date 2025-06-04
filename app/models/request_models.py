from pydantic import BaseModel

# 사용자가 보내는 질문 요청 모델 정의
class QueryWithArtworkId(BaseModel):
    artworkId: int    # 예: 42
    question: str     # 사용자 질문 내용
