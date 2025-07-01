from sentence_transformers import SentenceTransformer, util

# ✅ 모델 로딩 (최초 1회)
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# ✅ 의미 유사도 계산 함수
def get_similarity_score(query: str, content: str) -> float:
    query_embedding = model.encode(f"query: {query}", convert_to_tensor=True)
    content_embedding = model.encode(f"passage: {content}", convert_to_tensor=True)
    similarity = util.cos_sim(query_embedding, content_embedding)
    return similarity.item()  # float 반환