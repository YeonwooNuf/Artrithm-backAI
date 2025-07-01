from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import artchat                    # 작품 PDF 챗봇 라우터
from app.routes import recommend_faiss

# ✅ FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 라우터 등록

app.include_router(artchat.router, prefix="/api/artchat")                 # 작품 PDF 챗봇

app.include_router(recommend_faiss.router, prefix="/api")