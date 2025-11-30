"""
配置管理模块
使用pydantic-settings管理环境变量和应用配置
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类"""
    
    # API密钥
    alpha_vantage_api_key: str = Field(default="demo", alias="ALPHA_VANTAGE_API_KEY")
    finnhub_api_key: str = Field(default="", alias="FINNHUB_API_KEY")
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    
    # Telegram配置
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    
    # 数据库配置
    database_url: str = Field(default="sqlite:///./data/trading.db", alias="DATABASE_URL")
    
    # 应用配置
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # 数据采集配置
    data_fetch_interval: int = 60  # 数据采集间隔（秒）
    max_news_items: int = 20  # 最大新闻条数
    history_days: int = 365  # 历史数据天数
    
    # AI上下文配置
    context_max_records: int = 100  # 上下文最大记录数
    context_summary_interval: int = 10  # 每N条记录生成一次摘要
    
    # 代理配置（可选，用于访问Yahoo Finance等国外服务）
    http_proxy: Optional[str] = Field(default=None, alias="HTTP_PROXY")
    https_proxy: Optional[str] = Field(default=None, alias="HTTPS_PROXY")
    
    # ==================== v2.0 新增配置 ====================
    
    # 邮件配置 (QQ邮箱)
    email_smtp_host: str = Field(default="smtp.qq.com", alias="EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(default=465, alias="EMAIL_SMTP_PORT")
    email_sender: str = Field(default="", alias="EMAIL_SENDER")
    email_password: str = Field(default="", alias="EMAIL_PASSWORD")  # QQ邮箱授权码
    email_default_recipient: str = Field(default="", alias="EMAIL_DEFAULT_RECIPIENT")
    
    # 实时监控配置
    realtime_interval: int = Field(default=60, alias="REALTIME_INTERVAL")  # 刷新间隔（秒），30-300
    realtime_enabled: bool = Field(default=True, alias="REALTIME_ENABLED")
    default_history_years: int = Field(default=2, alias="DEFAULT_HISTORY_YEARS")  # 默认拉取历史数据年数
    
    # 预警配置
    alert_cooldown_minutes: int = Field(default=30, alias="ALERT_COOLDOWN_MINUTES")  # 预警冷却时间（分钟）
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def data_dir(self) -> Path:
        """数据目录"""
        path = Path("./data")
        path.mkdir(exist_ok=True)
        return path
    
    @property
    def logs_dir(self) -> Path:
        """日志目录"""
        path = Path("./logs")
        path.mkdir(exist_ok=True)
        return path


# 全局配置实例
settings = Settings()

