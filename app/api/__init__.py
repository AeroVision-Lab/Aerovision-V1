"""
API 模块
"""

from fastapi import APIRouter

from .routes import review, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(review.router, prefix="/review", tags=["review"])

__all__ = ["api_router"]
