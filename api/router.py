from fastapi import APIRouter

from api.routes import mcp_router

api_router = APIRouter()
api_router.include_router(mcp_router.router)