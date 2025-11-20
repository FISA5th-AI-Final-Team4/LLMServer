from pydantic import BaseModel
from typing import Optional
from enum import Enum


# --- 기본 요청/응답 스키마 ---
class QueryRequest(BaseModel):
    """쿼리 요청"""
    query: str

class QueryResponse(BaseModel):
    """쿼리 응답"""
    answer: str