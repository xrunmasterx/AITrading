"""
AI上下文管理器
负责管理和维护每只股票的分析上下文，供AI模型读取
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from loguru import logger

from app.config import settings
from app.models.analysis import AIContext, AnalysisRecord
from app.models.stock import StockInfo
from app.database.schemas import AIContextDB


class AIContextManager:
    """AI上下文管理器"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.max_records = settings.context_max_records
    
    def get_context(self, symbol: str) -> Optional[AIContext]:
        """获取股票的AI上下文"""
        db_context = self.db.query(AIContextDB).filter(
            AIContextDB.symbol == symbol
        ).first()
        
        if not db_context:
            return None
        
        return AIContext(
            symbol=db_context.symbol,
            created_at=db_context.created_at,
            updated_at=db_context.updated_at,
            stock_info_summary=db_context.stock_info_summary or "",
            recent_analyses=db_context.recent_analyses or [],
            key_events_timeline=db_context.key_events_timeline or [],
            analysis_count=db_context.analysis_count,
            avg_sentiment=db_context.avg_sentiment,
            price_trend=db_context.price_trend or "",
            user_notes=db_context.user_notes or []
        )
    
    def create_context(self, symbol: str, stock_info: Optional[StockInfo] = None) -> AIContext:
        """创建新的AI上下文"""
        context = AIContext(symbol=symbol)
        
        if stock_info:
            context.stock_info_summary = stock_info.to_context_str()
        
        # 保存到数据库
        db_context = AIContextDB(
            symbol=symbol,
            stock_info_summary=context.stock_info_summary,
            recent_analyses=[],
            key_events_timeline=[],
            analysis_count=0,
            user_notes=[]
        )
        self.db.add(db_context)
        self.db.commit()
        
        logger.info(f"创建AI上下文: {symbol}")
        return context
    
    def get_or_create_context(self, symbol: str, stock_info: Optional[StockInfo] = None) -> AIContext:
        """获取或创建AI上下文"""
        context = self.get_context(symbol)
        if context:
            return context
        return self.create_context(symbol, stock_info)
    
    def add_analysis(self, symbol: str, record: AnalysisRecord) -> AIContext:
        """添加分析记录到上下文"""
        context = self.get_or_create_context(symbol)
        context.add_analysis(record, self.max_records)
        
        # 更新数据库
        db_context = self.db.query(AIContextDB).filter(
            AIContextDB.symbol == symbol
        ).first()
        
        if db_context:
            db_context.recent_analyses = context.recent_analyses
            db_context.key_events_timeline = context.key_events_timeline
            db_context.analysis_count = context.analysis_count
            db_context.updated_at = datetime.now()
            
            # 更新平均情绪
            if context.recent_analyses:
                sentiments = [
                    a.get('sentiment') for a in context.recent_analyses 
                    if a.get('sentiment') is not None
                ]
                if sentiments:
                    db_context.avg_sentiment = sum(sentiments) / len(sentiments)
            
            self.db.commit()
        
        logger.debug(f"更新AI上下文: {symbol}, 分析次数: {context.analysis_count}")
        return context
    
    def update_stock_info(self, symbol: str, stock_info: StockInfo) -> None:
        """更新股票基本信息摘要"""
        db_context = self.db.query(AIContextDB).filter(
            AIContextDB.symbol == symbol
        ).first()
        
        if db_context:
            db_context.stock_info_summary = stock_info.to_context_str()
            db_context.updated_at = datetime.now()
            self.db.commit()
            logger.debug(f"更新股票信息摘要: {symbol}")
    
    def add_user_note(self, symbol: str, note: str) -> None:
        """添加用户备注"""
        db_context = self.db.query(AIContextDB).filter(
            AIContextDB.symbol == symbol
        ).first()
        
        if db_context:
            notes = db_context.user_notes or []
            notes.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}")
            db_context.user_notes = notes[-20:]  # 保留最近20条
            db_context.updated_at = datetime.now()
            self.db.commit()
            logger.info(f"添加用户备注: {symbol}")
    
    def add_key_event(self, symbol: str, event: str) -> None:
        """添加关键事件"""
        db_context = self.db.query(AIContextDB).filter(
            AIContextDB.symbol == symbol
        ).first()
        
        if db_context:
            events = db_context.key_events_timeline or []
            events.append({
                "timestamp": datetime.now().isoformat(),
                "event": event
            })
            db_context.key_events_timeline = events[-50:]  # 保留最近50条
            db_context.updated_at = datetime.now()
            self.db.commit()
            logger.info(f"添加关键事件: {symbol} - {event}")
    
    def get_prompt_context(self, symbol: str) -> str:
        """获取供AI使用的上下文提示词"""
        context = self.get_context(symbol)
        if not context:
            return f"暂无 {symbol} 的历史分析数据。"
        return context.to_prompt_context()
    
    def export_context(self, symbol: str) -> Optional[str]:
        """导出上下文为JSON"""
        context = self.get_context(symbol)
        if context:
            return context.to_json()
        return None
    
    def clear_context(self, symbol: str) -> bool:
        """清除股票的AI上下文"""
        db_context = self.db.query(AIContextDB).filter(
            AIContextDB.symbol == symbol
        ).first()
        
        if db_context:
            self.db.delete(db_context)
            self.db.commit()
            logger.warning(f"清除AI上下文: {symbol}")
            return True
        return False



