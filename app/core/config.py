"""
应用配置管理

通过环境变量和 Pydantic Settings 管理配置
"""

import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # 基础配置
    app_name: str = Field(default="AeroVision-V1", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # 服务配置
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=4, alias="WORKERS")

    # 日志配置
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT"
    )

    # 模型配置
    model_dir: Path = Field(default=Path("models"), alias="MODEL_DIR")
    device: str = Field(default="cuda:0", alias="DEVICE")

    # 审核阈值
    quality_threshold: float = Field(default=0.70, alias="QUALITY_THRESHOLD")
    registration_clarity_threshold: float = Field(
        default=0.80,
        alias="REGISTRATION_CLARITY_THRESHOLD"
    )
    occlusion_threshold: float = Field(default=0.20, alias="OCCLUSION_THRESHOLD")
    classifier_confidence_threshold: float = Field(
        default=0.50,
        alias="CLASSIFIER_CONFIDENCE_THRESHOLD"
    )

    # CORS 配置
    cors_origins: List[str] = Field(
        default=["*"],
        alias="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], alias="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], alias="CORS_ALLOW_HEADERS")

    # 请求限制
    max_image_size: int = Field(default=10 * 1024 * 1024, alias="MAX_IMAGE_SIZE")  # 10MB
    request_timeout: int = Field(default=60, alias="REQUEST_TIMEOUT")  # 60s

    # 外部服务
    backend_callback_url: Optional[str] = Field(default=None, alias="BACKEND_CALLBACK_URL")

    @property
    def model_dir_path(self) -> Path:
        """获取模型目录绝对路径"""
        if self.model_dir.is_absolute():
            return self.model_dir
        return Path.cwd() / self.model_dir


@lru_cache()
def get_settings() -> Settings:
    """获取缓存的配置实例"""
    return Settings()


# 便捷访问
settings = get_settings()
