from fastapi import APIRouter
from app.api.v1 import text
from app.api.v1 import image # assuming exists

api_router = APIRouter()
api_router.include_router(text.router, prefix="/v1")
api_router.include_router(image.router, prefix="/v1")
