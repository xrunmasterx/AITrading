"""
股票相关数据模型
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class StockInfo(BaseModel):
    """股票基本信息"""
    symbol: str = Field(..., description="股票代码，如AAPL, 0700.HK")
    name: str = Field(default="", description="股票名称")
    market: str = Field(default="", description="市场：US/HK")
    sector: str = Field(default="", description="行业板块")
    industry: str = Field(default="", description="细分行业")
    market_cap: Optional[float] = Field(default=None, description="市值")
    pe_ratio: Optional[float] = Field(default=None, description="市盈率")
    pb_ratio: Optional[float] = Field(default=None, description="市净率")
    dividend_yield: Optional[float] = Field(default=None, description="股息率")
    description: str = Field(default="", description="公司简介")
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def to_context_str(self) -> str:
        """转换为AI上下文字符串"""
        return (
            f"股票: {self.symbol} ({self.name})\n"
            f"市场: {self.market} | 行业: {self.sector}/{self.industry}\n"
            f"市值: {self.format_market_cap()} | PE: {self.pe_ratio or 'N/A'} | PB: {self.pb_ratio or 'N/A'}\n"
        )
    
    def format_market_cap(self) -> str:
        """格式化市值"""
        if not self.market_cap:
            return "N/A"
        if self.market_cap >= 1e12:
            return f"{self.market_cap/1e12:.2f}T"
        if self.market_cap >= 1e9:
            return f"{self.market_cap/1e9:.2f}B"
        if self.market_cap >= 1e6:
            return f"{self.market_cap/1e6:.2f}M"
        return f"{self.market_cap:.0f}"


class StockPrice(BaseModel):
    """股票价格数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None
    
    # 计算指标
    change: Optional[float] = None  # 涨跌额
    change_percent: Optional[float] = None  # 涨跌幅
    
    def calculate_change(self, prev_close: float) -> None:
        """计算涨跌幅"""
        if prev_close and prev_close > 0:
            self.change = self.close - prev_close
            self.change_percent = (self.change / prev_close) * 100


class StockNews(BaseModel):
    """股票新闻/舆情"""
    symbol: str
    title: str
    summary: str = ""
    source: str = ""
    url: str = ""
    published_at: datetime
    sentiment: Optional[str] = None  # positive/negative/neutral
    sentiment_score: Optional[float] = None  # -1 到 1
    relevance: Optional[float] = None  # 相关性评分
    
    def to_context_str(self) -> str:
        """转换为AI上下文字符串"""
        sentiment_str = f"[{self.sentiment}]" if self.sentiment else ""
        return f"[{self.published_at.strftime('%Y-%m-%d %H:%M')}] {sentiment_str} {self.title}"


class MarketOverview(BaseModel):
    """市场概览"""
    symbol: str
    current_price: float
    prev_close: float
    open_price: float
    day_high: float
    day_low: float
    volume: int
    avg_volume: Optional[int] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    
    @property
    def change(self) -> float:
        return self.current_price - self.prev_close
    
    @property
    def change_percent(self) -> float:
        if self.prev_close > 0:
            return (self.change / self.prev_close) * 100
        return 0.0



