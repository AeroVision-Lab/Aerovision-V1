"""
API 依赖注入

请求计数
"""

import threading
from datetime import datetime


class RequestCounter:
    """线程安全的请求计数器"""

    def __init__(self):
        self._lock = threading.Lock()
        self._total = 0
        self._success = 0
        self._failed = 0
        self._start_time = datetime.now()

    def increment(self, success: bool = True):
        """增加计数"""
        with self._lock:
            self._total += 1
            if success:
                self._success += 1
            else:
                self._failed += 1

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()
            return {
                "total_requests": self._total,
                "success_requests": self._success,
                "failed_requests": self._failed,
                "uptime_seconds": uptime,
                "requests_per_minute": (self._total / uptime * 60) if uptime > 0 else 0,
            }

    def reset(self):
        """重置计数器"""
        with self._lock:
            self._total = 0
            self._success = 0
            self._failed = 0
            self._start_time = datetime.now()


# 全局计数器实例
request_counter = RequestCounter()
