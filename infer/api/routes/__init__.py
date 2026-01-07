"""
API 路由模块
"""

from .review import router as review_router
from .health import router as health_router
from .admin import router as admin_router
from .stats import router as stats_router

__all__ = [
    "review_router",
    "health_router",
    "admin_router",
    "stats_router",
]
