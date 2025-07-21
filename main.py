from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import artchat                    # 작품 PDF 챗봇 라우터
from app.routes import recommend_faiss

# ✅ FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.0.56:5173",  # 내부 IP
        "http://172.30.1.54:5173",
        "http://192.168.10.159:5173",
        "http://localhost:5173",     # 로컬호스트도 추가
        "http://172.30.1.11:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 라우터 등록

app.include_router(artchat.router, prefix="/api/artchat")                 # 작품 PDF 챗봇

app.include_router(recommend_faiss.router, prefix="/api")