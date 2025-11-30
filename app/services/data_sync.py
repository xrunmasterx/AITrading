"""
智能数据同步管理器
负责检测数据状态并执行增量同步，避免冗余API调用
"""

from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from app.config import settings
from app.database.schemas import WatchlistDB, StockPriceDB, StockInfoDB
from app.models.stock import StockPrice, StockInfo


class SyncStatus(Enum):
    """同步状态"""
    NO_DATA = "no_data"          # 无数据，需全量拉取
    OUTDATED = "outdated"        # 过期，需增量更新
    UP_TO_DATE = "up_to_date"    # 最新，无需更新
    PARTIAL = "partial"          # 部分数据缺失


@dataclass
class SyncResult:
    """同步结果"""
    status: SyncStatus
    symbol: str
    records_added: int = 0
    records_updated: int = 0
    last_date: Optional[date] = None
    message: str = ""
    from_cache: bool = False


class DataSyncManager:
    """
    智能数据同步管理器
    
    核心职责：
    1. 检查本地数据状态
    2. 决定是否需要API调用
    3. 执行增量数据同步
    4. 更新同步时间戳
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.default_history_years = settings.default_history_years
    
    def check_data_status(self, symbol: str) -> Tuple[SyncStatus, Optional[date]]:
        """
        检查股票数据状态
        
        Returns:
            (SyncStatus, last_date): 状态和最后一条数据的日期
        """
        # 查询最新价格数据
        latest_price = self.db.query(StockPriceDB).filter(
            StockPriceDB.symbol == symbol,
            StockPriceDB.data_type == 'daily'
        ).order_by(StockPriceDB.timestamp.desc()).first()
        
        if not latest_price:
            logger.info(f"{symbol}: 无本地数据，需全量拉取")
            return SyncStatus.NO_DATA, None
        
        last_date = latest_price.timestamp.date() if latest_price.timestamp else None
        today = datetime.now().date()
        
        # 检查数据是否过期（超过1个交易日）
        # 考虑周末和节假日，使用2天作为阈值
        days_diff = (today - last_date).days if last_date else 999
        
        if days_diff <= 1:
            # 今天或昨天的数据，视为最新
            logger.debug(f"{symbol}: 数据最新 (最后日期: {last_date})")
            return SyncStatus.UP_TO_DATE, last_date
        elif days_diff <= 7:
            # 1-7天内，需要增量更新
            logger.info(f"{symbol}: 数据过期 {days_diff} 天，需增量更新")
            return SyncStatus.OUTDATED, last_date
        else:
            # 超过7天，建议全量刷新
            logger.info(f"{symbol}: 数据严重过期 ({days_diff} 天)，建议全量刷新")
            return SyncStatus.PARTIAL, last_date
    
    def get_last_price_date(self, symbol: str) -> Optional[date]:
        """获取本地最新价格日期"""
        latest = self.db.query(func.max(StockPriceDB.timestamp)).filter(
            StockPriceDB.symbol == symbol,
            StockPriceDB.data_type == 'daily'
        ).scalar()
        
        if latest:
            return latest.date() if isinstance(latest, datetime) else latest
        return None
    
    def get_price_count(self, symbol: str) -> int:
        """获取本地价格数据条数"""
        return self.db.query(StockPriceDB).filter(
            StockPriceDB.symbol == symbol,
            StockPriceDB.data_type == 'daily'
        ).count()
    
    def get_local_prices(
        self, 
        symbol: str, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[StockPrice]:
        """
        从本地数据库获取价格数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            limit: 最大条数
        """
        query = self.db.query(StockPriceDB).filter(
            StockPriceDB.symbol == symbol,
            StockPriceDB.data_type == 'daily'
        )
        
        if start_date:
            query = query.filter(StockPriceDB.timestamp >= datetime.combine(start_date, datetime.min.time()))
        if end_date:
            query = query.filter(StockPriceDB.timestamp <= datetime.combine(end_date, datetime.max.time()))
        
        query = query.order_by(StockPriceDB.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        results = query.all()
        
        return [
            StockPrice(
                symbol=r.symbol,
                timestamp=r.timestamp,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                adj_close=r.adj_close,
                change=r.change,
                change_percent=r.change_percent
            )
            for r in reversed(results)  # 按时间正序返回
        ]
    
    def save_prices(self, prices: List[StockPrice], data_type: str = 'daily') -> int:
        """
        保存价格数据到数据库（去重）
        
        Args:
            prices: 价格数据列表
            data_type: 数据类型 'daily' | 'intraday'
            
        Returns:
            新增记录数
        """
        if not prices:
            return 0
        
        symbol = prices[0].symbol
        added = 0
        
        for price in prices:
            # 检查是否已存在
            existing = self.db.query(StockPriceDB).filter(
                StockPriceDB.symbol == price.symbol,
                StockPriceDB.timestamp == price.timestamp,
                StockPriceDB.data_type == data_type
            ).first()
            
            if not existing:
                db_price = StockPriceDB(
                    symbol=price.symbol,
                    timestamp=price.timestamp,
                    open=price.open,
                    high=price.high,
                    low=price.low,
                    close=price.close,
                    volume=price.volume,
                    adj_close=price.adj_close,
                    change=price.change,
                    change_percent=price.change_percent,
                    data_type=data_type
                )
                self.db.add(db_price)
                added += 1
        
        if added > 0:
            self.db.commit()
            logger.info(f"保存 {symbol} 价格数据: {added} 条新增")
        
        return added
    
    def update_watchlist_sync_time(
        self, 
        symbol: str, 
        full_sync: bool = False,
        price_sync: bool = False
    ):
        """更新自选股同步时间戳"""
        watchlist = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == symbol
        ).first()
        
        if watchlist:
            now = datetime.now()
            if full_sync:
                watchlist.last_full_sync = now
            if price_sync:
                watchlist.last_price_sync = now
            self.db.commit()
    
    def get_watchlist_sync_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取自选股同步信息"""
        watchlist = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == symbol
        ).first()
        
        if not watchlist:
            return None
        
        return {
            'symbol': watchlist.symbol,
            'name': watchlist.name,
            'last_full_sync': watchlist.last_full_sync,
            'last_price_sync': watchlist.last_price_sync,
            'is_active': watchlist.is_active
        }
    
    def need_full_sync(self, symbol: str) -> bool:
        """判断是否需要全量同步"""
        watchlist = self.db.query(WatchlistDB).filter(
            WatchlistDB.symbol == symbol
        ).first()
        
        if not watchlist:
            return True
        
        if not watchlist.last_full_sync:
            return True
        
        # 超过1天未全量同步
        hours_since_sync = (datetime.now() - watchlist.last_full_sync).total_seconds() / 3600
        return hours_since_sync > 24
    
    def calculate_missing_date_range(self, symbol: str) -> Tuple[Optional[date], Optional[date]]:
        """
        计算需要补全的日期范围
        
        Returns:
            (start_date, end_date): 需要拉取的日期范围
        """
        last_date = self.get_last_price_date(symbol)
        today = datetime.now().date()
        
        if not last_date:
            # 无数据，从2年前开始
            start_date = today - timedelta(days=365 * self.default_history_years)
            return start_date, today
        
        # 有数据，从最后日期的下一天开始
        start_date = last_date + timedelta(days=1)
        
        if start_date >= today:
            # 已经是最新的
            return None, None
        
        return start_date, today
    
    def get_stock_info_from_db(self, symbol: str) -> Optional[StockInfo]:
        """从数据库获取股票信息"""
        info = self.db.query(StockInfoDB).filter(
            StockInfoDB.symbol == symbol
        ).first()
        
        if not info:
            return None
        
        return StockInfo(
            symbol=info.symbol,
            name=info.name,
            market=info.market,
            sector=info.sector,
            industry=info.industry,
            market_cap=info.market_cap,
            pe_ratio=info.pe_ratio,
            pb_ratio=info.pb_ratio,
            dividend_yield=info.dividend_yield,
            description=info.description
        )
    
    def save_stock_info(self, info: StockInfo) -> bool:
        """保存或更新股票信息"""
        existing = self.db.query(StockInfoDB).filter(
            StockInfoDB.symbol == info.symbol
        ).first()
        
        if existing:
            # 更新
            existing.name = info.name
            existing.market = info.market
            existing.sector = info.sector
            existing.industry = info.industry
            existing.market_cap = info.market_cap
            existing.pe_ratio = info.pe_ratio
            existing.pb_ratio = info.pb_ratio
            existing.dividend_yield = info.dividend_yield
            existing.description = info.description
            existing.updated_at = datetime.now()
        else:
            # 新增
            db_info = StockInfoDB(
                symbol=info.symbol,
                name=info.name,
                market=info.market,
                sector=info.sector,
                industry=info.industry,
                market_cap=info.market_cap,
                pe_ratio=info.pe_ratio,
                pb_ratio=info.pb_ratio,
                dividend_yield=info.dividend_yield,
                description=info.description
            )
            self.db.add(db_info)
        
        self.db.commit()
        return True
    
    def cleanup_old_intraday_data(self, symbol: str, keep_days: int = 7):
        """清理旧的分钟级数据"""
        cutoff = datetime.now() - timedelta(days=keep_days)
        
        deleted = self.db.query(StockPriceDB).filter(
            StockPriceDB.symbol == symbol,
            StockPriceDB.data_type == 'intraday',
            StockPriceDB.timestamp < cutoff
        ).delete()
        
        if deleted > 0:
            self.db.commit()
            logger.info(f"清理 {symbol} 旧分钟数据: {deleted} 条")
        
        return deleted

