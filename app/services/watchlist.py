"""
自选股管理服务
负责自选股列表的增删改查和激活切换
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from app.database.schemas import WatchlistDB, StockInfoDB
from app.utils.helpers import parse_symbol


class WatchlistService:
    """
    自选股管理服务
    
    功能：
    1. 添加/删除自选股
    2. 获取自选股列表
    3. 激活/切换股票
    4. 更新同步时间
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def add_stock(self, symbol: str, name: str = "", market: str = "") -> Optional[WatchlistDB]:
        """
        添加股票到自选股列表
        
        Args:
            symbol: 股票代码
            name: 股票名称（可选）
            market: 市场（可选，自动解析）
            
        Returns:
            新增的自选股记录，如果已存在返回现有记录
        """
        std_symbol, detected_market = parse_symbol(symbol)
        market = market or detected_market
        
        # 检查是否已存在
        existing = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == std_symbol
        ).first()
        
        if existing:
            logger.info(f"股票 {std_symbol} 已在自选股列表中")
            return existing
        
        # 尝试从 StockInfoDB 获取名称
        if not name:
            stock_info = self.db.query(StockInfoDB).filter(
                StockInfoDB.symbol == std_symbol
            ).first()
            if stock_info:
                name = stock_info.name
        
        # 创建新记录
        watchlist = WatchlistDB(
            symbol=std_symbol,
            name=name,
            market=market,
            added_at=datetime.now(),
            is_active=False,
            sort_order=self._get_next_sort_order()
        )
        
        self.db.add(watchlist)
        self.db.commit()
        
        logger.info(f"添加自选股: {std_symbol} ({name})")
        return watchlist
    
    def remove_stock(self, symbol: str) -> bool:
        """
        从自选股列表删除股票
        
        Args:
            symbol: 股票代码
            
        Returns:
            是否删除成功
        """
        std_symbol, _ = parse_symbol(symbol)
        
        deleted = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == std_symbol
        ).delete()
        
        self.db.commit()
        
        if deleted > 0:
            logger.info(f"删除自选股: {std_symbol}")
            return True
        
        return False
    
    def get_all(self) -> List[WatchlistDB]:
        """
        获取所有自选股
        
        Returns:
            按排序顺序返回的自选股列表
        """
        return self.db.query(WatchlistDB).order_by(
            WatchlistDB.sort_order.asc(),
            WatchlistDB.added_at.desc()
        ).all()
    
    def get_active(self) -> Optional[WatchlistDB]:
        """
        获取当前激活的股票
        
        Returns:
            激活的自选股，如果没有返回 None
        """
        return self.db.query(WatchlistDB).filter(
            WatchlistDB.is_active == True
        ).first()
    
    def set_active(self, symbol: str) -> Optional[WatchlistDB]:
        """
        设置股票为激活状态
        
        会自动取消其他股票的激活状态
        
        Args:
            symbol: 股票代码
            
        Returns:
            激活的自选股记录
        """
        std_symbol, _ = parse_symbol(symbol)
        
        # 先取消所有激活状态
        self.db.query(WatchlistDB).update({WatchlistDB.is_active: False})
        
        # 激活指定股票
        stock = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == std_symbol
        ).first()
        
        if stock:
            stock.is_active = True
            self.db.commit()
            logger.info(f"激活股票: {std_symbol}")
            return stock
        
        return None
    
    def deactivate_all(self):
        """取消所有股票的激活状态"""
        self.db.query(WatchlistDB).update({WatchlistDB.is_active: False})
        self.db.commit()
    
    def update_name(self, symbol: str, name: str) -> bool:
        """
        更新股票名称
        
        Args:
            symbol: 股票代码
            name: 新名称
        """
        std_symbol, _ = parse_symbol(symbol)
        
        stock = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == std_symbol
        ).first()
        
        if stock:
            stock.name = name
            self.db.commit()
            return True
        
        return False
    
    def update_sync_time(
        self, 
        symbol: str, 
        full_sync: bool = False, 
        price_sync: bool = False
    ):
        """
        更新同步时间戳
        
        Args:
            symbol: 股票代码
            full_sync: 是否更新全量同步时间
            price_sync: 是否更新价格同步时间
        """
        std_symbol, _ = parse_symbol(symbol)
        
        stock = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == std_symbol
        ).first()
        
        if stock:
            now = datetime.now()
            if full_sync:
                stock.last_full_sync = now
            if price_sync:
                stock.last_price_sync = now
            self.db.commit()
    
    def get_stock(self, symbol: str) -> Optional[WatchlistDB]:
        """
        获取指定股票
        
        Args:
            symbol: 股票代码
        """
        std_symbol, _ = parse_symbol(symbol)
        
        return self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == std_symbol
        ).first()
    
    def exists(self, symbol: str) -> bool:
        """
        检查股票是否在自选股列表中
        
        Args:
            symbol: 股票代码
        """
        std_symbol, _ = parse_symbol(symbol)
        
        return self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == std_symbol
        ).first() is not None
    
    def get_count(self) -> int:
        """获取自选股数量"""
        return self.db.query(WatchlistDB).count()
    
    def reorder(self, symbols: List[str]):
        """
        重新排序自选股
        
        Args:
            symbols: 按新顺序排列的股票代码列表
        """
        for i, symbol in enumerate(symbols):
            std_symbol, _ = parse_symbol(symbol)
            stock = self.db.query(WatchlistDB).filter(
                WatchlistDB.symbol == std_symbol
            ).first()
            if stock:
                stock.sort_order = i
        
        self.db.commit()
    
    def _get_next_sort_order(self) -> int:
        """获取下一个排序号"""
        max_order = self.db.query(func.max(WatchlistDB.sort_order)).scalar()
        return (max_order or 0) + 1
    
    def get_watchlist_summary(self) -> List[Dict[str, Any]]:
        """
        获取自选股摘要列表（用于UI显示）
        
        Returns:
            [{'symbol': 'BABA', 'name': '阿里巴巴', 'is_active': True, ...}, ...]
        """
        stocks = self.get_all()
        
        return [
            {
                'symbol': s.symbol,
                'name': s.name,
                'market': s.market,
                'is_active': s.is_active,
                'added_at': s.added_at,
                'last_full_sync': s.last_full_sync,
                'last_price_sync': s.last_price_sync,
                'needs_sync': self._needs_sync(s)
            }
            for s in stocks
        ]
    
    def _needs_sync(self, stock: WatchlistDB) -> bool:
        """判断是否需要同步"""
        if not stock.last_full_sync:
            return True
        
        hours_since = (datetime.now() - stock.last_full_sync).total_seconds() / 3600
        return hours_since > 24


# 便捷函数
def create_watchlist_service(db_session: Session) -> WatchlistService:
    """创建自选股服务实例"""
    return WatchlistService(db_session)

