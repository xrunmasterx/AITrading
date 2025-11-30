"""
分析记录和AI上下文数据模型
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json


class TechnicalIndicators(BaseModel):
    """技术指标"""
    sma_5: Optional[float] = None   # 5日均线
    sma_10: Optional[float] = None  # 10日均线
    sma_20: Optional[float] = None  # 20日均线
    sma_50: Optional[float] = None  # 50日均线
    sma_200: Optional[float] = None # 200日均线
    rsi_14: Optional[float] = None  # 14日RSI
    macd: Optional[float] = None    # MACD
    macd_signal: Optional[float] = None  # MACD信号线
    macd_hist: Optional[float] = None    # MACD柱
    bollinger_upper: Optional[float] = None   # 布林带上轨
    bollinger_middle: Optional[float] = None  # 布林带中轨
    bollinger_lower: Optional[float] = None   # 布林带下轨
    atr: Optional[float] = None     # ATR
    
    def to_summary(self) -> str:
        """生成技术指标摘要"""
        parts = []
        if self.rsi_14:
            status = "超买" if self.rsi_14 > 70 else ("超卖" if self.rsi_14 < 30 else "中性")
            parts.append(f"RSI({self.rsi_14:.1f}, {status})")
        if self.sma_20 and self.sma_50:
            trend = "多头排列" if self.sma_20 > self.sma_50 else "空头排列"
            parts.append(f"均线{trend}")
        if self.macd and self.macd_signal:
            signal = "金叉" if self.macd > self.macd_signal else "死叉"
            parts.append(f"MACD{signal}")
        return " | ".join(parts) if parts else "无技术指标"


class AnalysisRecord(BaseModel):
    """分析记录"""
    id: Optional[int] = None
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # 价格信息
    current_price: float
    price_change: Optional[float] = None
    price_change_percent: Optional[float] = None
    volume: Optional[int] = None
    
    # 技术指标
    technical: Optional[TechnicalIndicators] = None
    
    # 新闻摘要
    news_summary: str = ""
    news_count: int = 0
    sentiment_score: Optional[float] = None  # 综合舆情得分
    sentiment_label: Optional[str] = None    # 情绪标签
    
    # 策略分析
    strategy_signal: Optional[str] = None      # 策略信号
    strategy_confidence: Optional[float] = None # 策略置信度
    
    # 关键事件标记
    key_events: List[str] = Field(default_factory=list)
    
    # AI生成的分析摘要
    ai_summary: str = ""
    
    def to_context_dict(self) -> Dict[str, Any]:
        """转换为AI上下文字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "price": self.current_price,
            "change_percent": self.price_change_percent,
            "technical_summary": self.technical.to_summary() if self.technical else "",
            "sentiment": self.sentiment_score,
            "key_events": self.key_events,
            "summary": self.ai_summary
        }


class AIContext(BaseModel):
    """AI上下文管理"""
    symbol: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # 股票基本信息摘要
    stock_info_summary: str = ""
    
    # 历史分析记录（最近N条）
    recent_analyses: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 关键事件时间线
    key_events_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    # 累计统计
    analysis_count: int = 0
    avg_sentiment: Optional[float] = None
    price_trend: str = ""  # 上涨/下跌/震荡
    
    # 用户备注
    user_notes: List[str] = Field(default_factory=list)
    
    def add_analysis(self, record: AnalysisRecord, max_records: int = 100) -> None:
        """添加分析记录到上下文"""
        self.recent_analyses.append(record.to_context_dict())
        if len(self.recent_analyses) > max_records:
            self.recent_analyses = self.recent_analyses[-max_records:]
        
        self.analysis_count += 1
        self.updated_at = datetime.now()
        
        # 更新关键事件
        for event in record.key_events:
            self.key_events_timeline.append({
                "timestamp": record.timestamp.isoformat(),
                "event": event
            })
    
    def to_prompt_context(self) -> str:
        """生成供AI使用的上下文字符串"""
        context_parts = [
            f"=== {self.symbol} 分析上下文 ===",
            f"更新时间: {self.updated_at.strftime('%Y-%m-%d %H:%M')}",
            f"累计分析: {self.analysis_count}次",
            "",
            "【基本信息】",
            self.stock_info_summary or "暂无",
            "",
            "【近期分析摘要】"
        ]
        
        # 添加最近5条分析
        for analysis in self.recent_analyses[-5:]:
            context_parts.append(
                f"- {analysis.get('timestamp', '')}: "
                f"价格{analysis.get('price', 'N/A')} "
                f"({analysis.get('change_percent', 0):+.2f}%) "
                f"| {analysis.get('technical_summary', '')}"
            )
        
        if self.key_events_timeline:
            context_parts.extend([
                "",
                "【关键事件】"
            ])
            for event in self.key_events_timeline[-10:]:
                context_parts.append(f"- {event['timestamp']}: {event['event']}")
        
        if self.user_notes:
            context_parts.extend([
                "",
                "【用户备注】"
            ])
            for note in self.user_notes[-5:]:
                context_parts.append(f"- {note}")
        
        return "\n".join(context_parts)
    
    def to_json(self) -> str:
        """导出为JSON"""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "AIContext":
        """从JSON导入"""
        return cls.model_validate_json(json_str)



