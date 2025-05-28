# FastAPI 서버를 실행하고, 위에서 만든 라우터를 등록하는 코드

# main.py
from fastapi import FastAPI
from app.routes import artchat

# FastAPI 애플리케이션 생성
app = FastAPI()

# artchat 관련 라우터 등록 (URL prefix: /api/artchat)
app.include_router(artchat.router, prefix="/api/artchat")