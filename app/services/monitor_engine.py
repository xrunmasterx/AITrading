"""
实时监控引擎
负责驱动实时数据获取、预警检查和UI刷新
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from loguru import logger

from app.config import settings
from app.services.data_fetcher import DataFetcher
from app.services.alert_checker import AlertChecker
from app.database.schemas import IntradayPriceDB


@dataclass
class PricePoint:
    """价格数据点"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    day_high: float = 0
    day_low: float = 0


@dataclass
class MonitorState:
    """监控状态"""
    symbol: str
    is_running: bool = False
    last_price: Optional[PricePoint] = None
    last_update: Optional[datetime] = None
    error_count: int = 0
    session_prices: List[PricePoint] = field(default_factory=list)


class RealtimeMonitorEngine:
    """
    实时监控引擎
    
    核心功能：
    1. 定时获取实时价格（使用轻量级 API）
    2. 维护当前会话的价格数据流
    3. 检查价格预警
    4. 触发 UI 刷新回调
    
    使用方式：
    ```python
    engine = RealtimeMonitorEngine()
    engine.set_ui_callback(update_ui_function)
    await engine.start_monitoring("BABA")
    ```
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.fetcher = DataFetcher()
        self.alert_checker = AlertChecker(db_session) if db_session else None
        
        # 状态管理
        self.state = MonitorState(symbol="")
        self.interval = settings.realtime_interval
        self._stop_event = asyncio.Event()
        
        # UI 回调函数
        self._ui_callback: Optional[Callable] = None
        self._on_price_update: Optional[Callable] = None
        self._on_alert_triggered: Optional[Callable] = None
    
    def set_interval(self, seconds: int):
        """
        设置刷新间隔
        
        Args:
            seconds: 刷新间隔（30-300秒）
        """
        self.interval = max(30, min(300, seconds))
        logger.info(f"监控刷新间隔设置为: {self.interval} 秒")
    
    def set_ui_callback(self, callback: Callable):
        """
        设置 UI 刷新回调
        
        Args:
            callback: 回调函数，接收 MonitorState 参数
        """
        self._ui_callback = callback
    
    def set_on_price_update(self, callback: Callable):
        """设置价格更新回调"""
        self._on_price_update = callback
    
    def set_on_alert_triggered(self, callback: Callable):
        """设置预警触发回调"""
        self._on_alert_triggered = callback
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self.state.is_running
    
    @property
    def active_symbol(self) -> Optional[str]:
        """当前监控的股票"""
        return self.state.symbol if self.state.is_running else None
    
    @property
    def last_price(self) -> Optional[PricePoint]:
        """最新价格"""
        return self.state.last_price
    
    @property
    def session_prices(self) -> List[PricePoint]:
        """当前会话的价格历史"""
        return self.state.session_prices
    
    async def start_monitoring(self, symbol: str) -> bool:
        """
        启动实时监控
        
        Args:
            symbol: 股票代码
            
        Returns:
            是否成功启动
        """
        if self.state.is_running:
            # 如果正在监控其他股票，先停止
            if self.state.symbol != symbol:
                await self.stop_monitoring()
            else:
                logger.warning(f"已经在监控 {symbol}")
                return True
        
        self.state = MonitorState(symbol=symbol)
        self.state.is_running = True
        self._stop_event.clear()
        
        logger.info(f"启动实时监控: {symbol}, 间隔: {self.interval}秒")
        
        # 启动监控循环（非阻塞）
        asyncio.create_task(self._monitor_loop())
        
        return True
    
    async def stop_monitoring(self):
        """停止实时监控"""
        if not self.state.is_running:
            return
        
        logger.info(f"停止实时监控: {self.state.symbol}")
        
        self.state.is_running = False
        self._stop_event.set()
        
        # 等待一小段时间确保循环退出
        await asyncio.sleep(0.1)
    
    async def _monitor_loop(self):
        """
        监控主循环
        
        每隔 interval 秒执行一次：
        1. 获取实时价格
        2. 更新会话数据
        3. 检查预警
        4. 触发 UI 刷新
        """
        symbol = self.state.symbol
        logger.info(f"监控循环启动: {symbol}")
        
        while self.state.is_running:
            try:
                # 获取实时价格
                snapshot = await self.fetcher.fetch_realtime_snapshot(symbol)
                
                if snapshot:
                    # 创建价格点
                    price_point = PricePoint(
                        symbol=symbol,
                        price=snapshot['price'],
                        change=snapshot['change'],
                        change_percent=snapshot['change_percent'],
                        volume=snapshot['volume'],
                        timestamp=snapshot['timestamp'],
                        day_high=snapshot.get('day_high', 0),
                        day_low=snapshot.get('day_low', 0)
                    )
                    
                    # 更新状态
                    self.state.last_price = price_point
                    self.state.last_update = datetime.now()
                    self.state.session_prices.append(price_point)
                    self.state.error_count = 0
                    
                    # 限制会话数据量（保留最近100个点）
                    if len(self.state.session_prices) > 100:
                        self.state.session_prices = self.state.session_prices[-100:]
                    
                    # 保存到数据库（可选）
                    if self.db:
                        await self._save_intraday_price(price_point)
                    
                    # 检查预警
                    if self.alert_checker:
                        alert_result = await self.alert_checker.check_and_notify(
                            symbol, 
                            price_point.price
                        )
                        if alert_result and self._on_alert_triggered:
                            self._on_alert_triggered(alert_result)
                    
                    # 触发价格更新回调
                    if self._on_price_update:
                        self._on_price_update(price_point)
                    
                    # 触发 UI 刷新
                    if self._ui_callback:
                        self._ui_callback(self.state)
                    
                    logger.debug(f"{symbol}: ${price_point.price:.2f} ({price_point.change_percent:+.2f}%)")
                else:
                    self.state.error_count += 1
                    if self.state.error_count >= 3:
                        logger.warning(f"{symbol}: 连续 {self.state.error_count} 次获取失败")
                
            except Exception as e:
                self.state.error_count += 1
                logger.error(f"监控循环异常: {e}")
            
            # 等待下一次循环
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), 
                    timeout=self.interval
                )
                # 如果 stop_event 被触发，退出循环
                break
            except asyncio.TimeoutError:
                # 正常超时，继续下一次循环
                pass
        
        logger.info(f"监控循环结束: {symbol}")
    
    async def _save_intraday_price(self, price: PricePoint):
        """保存分钟级价格到数据库"""
        if not self.db:
            return
        
        try:
            db_price = IntradayPriceDB(
                symbol=price.symbol,
                timestamp=price.timestamp,
                price=price.price,
                volume=price.volume,
                change=price.change,
                change_percent=price.change_percent,
                data_type='realtime'
            )
            self.db.add(db_price)
            self.db.commit()
        except Exception as e:
            logger.error(f"保存分钟数据失败: {e}")
            self.db.rollback()
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取监控状态
        
        Returns:
            {
                'symbol': str,
                'is_running': bool,
                'last_price': float,
                'last_update': datetime,
                'interval': int,
                'session_count': int,
                'error_count': int
            }
        """
        return {
            'symbol': self.state.symbol,
            'is_running': self.state.is_running,
            'last_price': self.state.last_price.price if self.state.last_price else None,
            'last_change': self.state.last_price.change if self.state.last_price else None,
            'last_change_percent': self.state.last_price.change_percent if self.state.last_price else None,
            'last_update': self.state.last_update,
            'interval': self.interval,
            'session_count': len(self.state.session_prices),
            'error_count': self.state.error_count,
            'next_update_in': self._get_next_update_seconds()
        }
    
    def _get_next_update_seconds(self) -> int:
        """计算距离下次更新的秒数"""
        if not self.state.is_running or not self.state.last_update:
            return 0
        
        elapsed = (datetime.now() - self.state.last_update).total_seconds()
        remaining = max(0, self.interval - elapsed)
        return int(remaining)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        获取当前会话的价格摘要
        
        Returns:
            {
                'start_price': float,
                'current_price': float,
                'session_high': float,
                'session_low': float,
                'price_count': int,
                'start_time': datetime
            }
        """
        prices = self.state.session_prices
        
        if not prices:
            return {}
        
        price_values = [p.price for p in prices]
        
        return {
            'start_price': prices[0].price,
            'current_price': prices[-1].price,
            'session_change': prices[-1].price - prices[0].price,
            'session_change_percent': ((prices[-1].price - prices[0].price) / prices[0].price * 100) if prices[0].price else 0,
            'session_high': max(price_values),
            'session_low': min(price_values),
            'price_count': len(prices),
            'start_time': prices[0].timestamp,
            'last_time': prices[-1].timestamp
        }
    
    def clear_session(self):
        """清除当前会话数据"""
        self.state.session_prices = []
        logger.info("会话数据已清除")


# 全局监控引擎实例
_monitor_engine: Optional[RealtimeMonitorEngine] = None


def get_monitor_engine(db_session=None) -> RealtimeMonitorEngine:
    """
    获取全局监控引擎实例
    
    Args:
        db_session: 数据库会话（首次调用时需要）
    """
    global _monitor_engine
    
    if _monitor_engine is None:
        _monitor_engine = RealtimeMonitorEngine(db_session)
    
    return _monitor_engine


def create_monitor_engine(db_session=None) -> RealtimeMonitorEngine:
    """创建新的监控引擎实例"""
    return RealtimeMonitorEngine(db_session)

