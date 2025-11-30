"""
通用工具函数
"""

from datetime import datetime, time
from typing import Tuple, Optional
import pytz


def parse_symbol(symbol: str) -> Tuple[str, str]:
    """
    解析股票代码，返回(标准化代码, 市场)
    
    Args:
        symbol: 股票代码，如 AAPL, 0700.HK, 00700
        
    Returns:
        (标准化代码, 市场): 如 ("AAPL", "US"), ("0700.HK", "HK")
    """
    symbol = symbol.upper().strip()
    
    # 港股判断
    if symbol.endswith(".HK"):
        return symbol, "HK"
    
    # 纯数字视为港股
    if symbol.isdigit():
        # 补齐4位
        code = symbol.zfill(4)
        return f"{code}.HK", "HK"
    
    # 其他视为美股
    return symbol, "US"


def format_number(value: Optional[float], decimals: int = 2) -> str:
    """格式化数字"""
    if value is None:
        return "N/A"
    
    if abs(value) >= 1e12:
        return f"{value/1e12:.{decimals}f}T"
    if abs(value) >= 1e9:
        return f"{value/1e9:.{decimals}f}B"
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    if abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    
    return f"{value:.{decimals}f}"


def format_currency(value: Optional[float], currency: str = "USD", decimals: int = 2) -> str:
    """格式化货币"""
    if value is None:
        return "N/A"
    
    symbols = {
        "USD": "$",
        "HKD": "HK$",
        "CNY": "¥"
    }
    symbol = symbols.get(currency, currency)
    return f"{symbol}{value:,.{decimals}f}"


def format_percent(value: Optional[float], decimals: int = 2) -> str:
    """格式化百分比"""
    if value is None:
        return "N/A"
    return f"{value:+.{decimals}f}%"


def get_market_status(market: str = "US") -> Tuple[str, str]:
    """
    获取市场状态
    
    Returns:
        (状态, 描述): 如 ("open", "交易中"), ("closed", "已休市")
    """
    now = datetime.now()
    
    if market == "US":
        # 美股交易时间: 9:30 AM - 4:00 PM ET (美东时间)
        et_tz = pytz.timezone("America/New_York")
        et_now = datetime.now(et_tz)
        
        # 周末休市
        if et_now.weekday() >= 5:
            return "closed", "周末休市"
        
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = et_now.time()
        
        if market_open <= current_time <= market_close:
            return "open", "交易中"
        elif current_time < market_open:
            return "pre_market", "盘前"
        else:
            return "after_hours", "盘后"
    
    elif market == "HK":
        # 港股交易时间: 9:30 AM - 12:00 PM, 1:00 PM - 4:00 PM HKT
        hk_tz = pytz.timezone("Asia/Hong_Kong")
        hk_now = datetime.now(hk_tz)
        
        # 周末休市
        if hk_now.weekday() >= 5:
            return "closed", "周末休市"
        
        morning_open = time(9, 30)
        morning_close = time(12, 0)
        afternoon_open = time(13, 0)
        afternoon_close = time(16, 0)
        current_time = hk_now.time()
        
        if (morning_open <= current_time <= morning_close or 
            afternoon_open <= current_time <= afternoon_close):
            return "open", "交易中"
        elif morning_close < current_time < afternoon_open:
            return "lunch_break", "午间休市"
        else:
            return "closed", "已休市"
    
    return "unknown", "未知市场"


def calculate_support_resistance(prices: list, window: int = 20) -> Tuple[float, float]:
    """
    简单计算支撑位和阻力位
    
    Args:
        prices: 价格列表
        window: 计算窗口
        
    Returns:
        (支撑位, 阻力位)
    """
    if not prices or len(prices) < window:
        return 0.0, 0.0
    
    recent_prices = prices[-window:]
    support = min(recent_prices)
    resistance = max(recent_prices)
    
    return support, resistance



