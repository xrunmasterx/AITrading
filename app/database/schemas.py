"""
数据库表结构定义
使用SQLAlchemy ORM
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class StockInfoDB(Base):
    """股票基本信息表"""
    __tablename__ = "stock_info"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), default="")
    market = Column(String(10), default="")  # US/HK
    sector = Column(String(100), default="")
    industry = Column(String(100), default="")
    market_cap = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    pb_ratio = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<StockInfo(symbol={self.symbol}, name={self.name})>"


class StockPriceDB(Base):
    """股票价格历史表"""
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adj_close = Column(Float, nullable=True)
    change = Column(Float, nullable=True)
    change_percent = Column(Float, nullable=True)
    data_type = Column(String(20), default="daily")  # 'daily' | 'intraday'
    
    # 复合索引优化查询
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_symbol_datatype', 'symbol', 'data_type'),
    )
    
    def __repr__(self):
        return f"<StockPrice(symbol={self.symbol}, date={self.timestamp}, close={self.close})>"


class StockNewsDB(Base):
    """股票新闻/舆情表"""
    __tablename__ = "stock_news"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text, default="")
    source = Column(String(100), default="")
    url = Column(String(1000), default="")
    published_at = Column(DateTime, nullable=False)
    sentiment = Column(String(20), nullable=True)  # positive/negative/neutral
    sentiment_score = Column(Float, nullable=True)
    relevance = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_news_symbol_date', 'symbol', 'published_at'),
    )
    
    def __repr__(self):
        return f"<StockNews(symbol={self.symbol}, title={self.title[:30]}...)>"


class AnalysisRecordDB(Base):
    """分析记录表"""
    __tablename__ = "analysis_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    
    # 价格信息
    current_price = Column(Float, nullable=False)
    price_change = Column(Float, nullable=True)
    price_change_percent = Column(Float, nullable=True)
    
    # 技术指标（JSON存储）
    technical_indicators = Column(JSON, nullable=True)
    
    # 新闻摘要
    news_summary = Column(Text, default="")
    sentiment_score = Column(Float, nullable=True)
    
    # 关键事件（JSON数组）
    key_events = Column(JSON, default=list)
    
    # AI分析摘要
    ai_summary = Column(Text, default="")
    
    __table_args__ = (
        Index('idx_analysis_symbol_time', 'symbol', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<AnalysisRecord(symbol={self.symbol}, time={self.timestamp})>"


class AIContextDB(Base):
    """AI上下文存储表"""
    __tablename__ = "ai_contexts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 股票信息摘要
    stock_info_summary = Column(Text, default="")
    
    # 历史分析记录（JSON数组）
    recent_analyses = Column(JSON, default=list)
    
    # 关键事件时间线（JSON数组）
    key_events_timeline = Column(JSON, default=list)
    
    # 统计信息
    analysis_count = Column(Integer, default=0)
    avg_sentiment = Column(Float, nullable=True)
    price_trend = Column(String(20), default="")
    
    # 用户备注（JSON数组）
    user_notes = Column(JSON, default=list)
    
    def __repr__(self):
        return f"<AIContext(symbol={self.symbol}, count={self.analysis_count})>"


class PriceAlertDB(Base):
    """价格预警表（旧版，保留兼容）"""
    __tablename__ = "price_alerts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    alert_type = Column(String(20), nullable=False)  # above/below/change_percent
    target_value = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    is_triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime, nullable=True)
    notification_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<PriceAlert(symbol={self.symbol}, type={self.alert_type}, target={self.target_value})>"


# ==================== v2.0 新增表 ====================

class WatchlistDB(Base):
    """自选股列表表"""
    __tablename__ = "watchlist"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), default="")
    market = Column(String(10), default="US")  # US/HK
    added_at = Column(DateTime, default=datetime.now)
    last_full_sync = Column(DateTime, nullable=True)  # 上次全量同步时间
    last_price_sync = Column(DateTime, nullable=True)  # 上次价格同步时间
    is_active = Column(Boolean, default=False)  # 当前激活状态
    sort_order = Column(Integer, default=0)  # 排序顺序
    
    def __repr__(self):
        return f"<Watchlist(symbol={self.symbol}, active={self.is_active})>"


class IntradayPriceDB(Base):
    """分钟级实时价格表"""
    __tablename__ = "intraday_prices"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=True)
    change = Column(Float, nullable=True)
    change_percent = Column(Float, nullable=True)
    data_type = Column(String(20), default="realtime")  # 'realtime' | 'session'
    
    __table_args__ = (
        Index('idx_intraday_symbol_time', 'symbol', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<IntradayPrice(symbol={self.symbol}, time={self.timestamp}, price={self.price})>"


class AlertConfigDB(Base):
    """价格预警配置表（v2.0增强版）"""
    __tablename__ = "alert_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    upper_limit = Column(Float, nullable=True)  # 价格上限
    lower_limit = Column(Float, nullable=True)  # 价格下限
    change_percent_threshold = Column(Float, nullable=True)  # 涨跌幅阈值
    email = Column(String(200), nullable=True)  # 接收邮箱
    is_enabled = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime, nullable=True)
    last_trigger_type = Column(String(20), nullable=True)  # 'upper' | 'lower' | 'change'
    cooldown_minutes = Column(Integer, default=30)  # 防轰炸冷却时间（分钟）
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    __table_args__ = (
        Index('idx_alert_symbol', 'symbol'),
    )
    
    def __repr__(self):
        return f"<AlertConfig(symbol={self.symbol}, upper={self.upper_limit}, lower={self.lower_limit})>"



