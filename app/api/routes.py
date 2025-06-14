from fastapi import APIRouter
from app.api.v1 import text

api_router = APIRouter()
api_router.include_router(text.router, prefix="/v1")
