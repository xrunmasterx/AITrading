"""
数据采集模块
支持从多个数据源获取股票数据
- yfinance: 主要数据源（免费，带重试机制）
- Finnhub: 新闻数据
- Alpha Vantage: 最后备用
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import yfinance as yf
import finnhub
import httpx
from loguru import logger

from app.config import settings
from app.models.stock import StockInfo, StockPrice, StockNews, MarketOverview
from app.utils.helpers import parse_symbol


class DataFetcher:
    """
    数据采集器 - yfinance 优先
    
    关键优化：
    1. 复用 Ticker 对象，避免重复创建
    2. 增加请求间隔，避免限速
    3. 本地文件缓存，减少 API 调用
    """
    
    # 类级别缓存
    _price_cache: Dict[str, Dict[str, Any]] = {}
    _info_cache: Dict[str, Dict[str, Any]] = {}
    _ticker_cache: Dict[str, Any] = {}  # 缓存 Ticker 对象
    _cache_ttl = 600  # 缓存有效期10分钟
    _long_cache_ttl = 3600  # 长期缓存1小时
    
    # 请求控制 - 关键参数
    _last_request_time: float = 0
    _request_interval: float = 1.5  # 请求间隔1.5秒（Yahoo建议不超过2000次/小时）
    _retry_count: int = 2  # 减少重试次数
    _retry_delay: float = 3.0  # 增加重试间隔
    _rate_limited: bool = False  # 是否被限速
    _rate_limit_until: float = 0  # 限速解除时间
    
    def __init__(self):
        # 配置代理（如果设置了）
        self._setup_proxy()
        
        # Finnhub客户端
        self.finnhub_client = None
        if settings.finnhub_api_key:
            self.finnhub_client = finnhub.Client(api_key=settings.finnhub_api_key)
        
        # HTTP客户端（带代理支持）
        proxy_url = settings.https_proxy or settings.http_proxy
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            proxy=proxy_url if proxy_url else None
        )
        
        self.alpha_vantage_base = "https://www.alphavantage.co/query"
    
    def _get_ticker(self, symbol: str) -> Any:
        """获取或创建 Ticker 对象（复用缓存）"""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]
    
    async def _rate_limit(self):
        """请求限速控制"""
        # 如果当前被限速，等待
        if self._rate_limited and time.time() < self._rate_limit_until:
            wait_time = self._rate_limit_until - time.time()
            logger.warning(f"API限速中，等待 {wait_time:.1f} 秒...")
            await asyncio.sleep(wait_time)
            self._rate_limited = False
        
        # 正常请求间隔
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._request_interval:
            await asyncio.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _mark_rate_limited(self, seconds: int = 60):
        """标记被限速"""
        self._rate_limited = True
        self._rate_limit_until = time.time() + seconds
        logger.warning(f"检测到API限速，暂停请求 {seconds} 秒")
    
    def _setup_proxy(self):
        """设置代理环境变量"""
        if settings.http_proxy:
            os.environ['HTTP_PROXY'] = settings.http_proxy
            os.environ['http_proxy'] = settings.http_proxy
        if settings.https_proxy:
            os.environ['HTTPS_PROXY'] = settings.https_proxy
            os.environ['https_proxy'] = settings.https_proxy
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    # ==================== 股票基本信息 ====================
    
    async def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """
        获取股票基本信息（带缓存和重试）
        
        Args:
            symbol: 股票代码，如 AAPL, 0700.HK
        """
        std_symbol, market = parse_symbol(symbol)
        
        # 检查缓存
        cached = self._get_info_from_cache(std_symbol)
        if cached:
            logger.info(f"使用缓存的股票信息: {std_symbol}")
            return cached
        
        # 尝试yfinance（带重试）
        info = await self._get_stock_info_yfinance_with_retry(std_symbol, market)
        
        # 如果yfinance彻底失败，最后尝试Alpha Vantage
        if not info:
            info = await self._get_stock_info_alpha_vantage(std_symbol, market)
        
        # 缓存成功获取的数据
        if info:
            self._save_info_to_cache(std_symbol, info)
        
        return info
    
    async def _get_stock_info_yfinance_with_retry(self, symbol: str, market: str) -> Optional[StockInfo]:
        """从yfinance获取股票信息（带重试机制）"""
        for attempt in range(self._retry_count):
            await self._rate_limit()  # 请求限速
            
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if not info or 'symbol' not in info:
                    logger.warning(f"yfinance未找到股票信息: {symbol}")
                    return None
                
                return StockInfo(
                    symbol=symbol,
                    name=info.get('longName') or info.get('shortName', ''),
                    market=market,
                    sector=info.get('sector', ''),
                    industry=info.get('industry', ''),
                    market_cap=info.get('marketCap'),
                    pe_ratio=info.get('trailingPE'),
                    pb_ratio=info.get('priceToBook'),
                    dividend_yield=info.get('dividendYield'),
                    description=info.get('longBusinessSummary', '')[:500] if info.get('longBusinessSummary') else ''
                )
            except Exception as e:
                error_msg = str(e)
                if 'RateLimit' in error_msg or 'Too Many Requests' in error_msg:
                    if attempt < self._retry_count - 1:
                        wait_time = self._retry_delay * (attempt + 1)
                        logger.warning(f"yfinance限速，{wait_time}秒后重试 ({attempt + 1}/{self._retry_count}): {symbol}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"yfinance限速，已达最大重试次数: {symbol}")
                else:
                    logger.error(f"yfinance获取股票信息失败: {symbol}, 错误: {e}")
                    break
        
        return None
    
    def _get_info_from_cache(self, symbol: str) -> Optional[StockInfo]:
        """从缓存获取股票信息"""
        if symbol not in self._info_cache:
            return None
        cached = self._info_cache[symbol]
        cache_time = cached.get('timestamp')
        if cache_time and (datetime.now() - cache_time).seconds < self._cache_ttl:
            return cached.get('info')
        # 即使过期，被限速时也返回旧数据
        return cached.get('info')
    
    def _save_info_to_cache(self, symbol: str, info: StockInfo):
        """保存股票信息到缓存"""
        self._info_cache[symbol] = {
            'info': info,
            'timestamp': datetime.now()
        }
    
    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        """安全地将值转换为浮点数，处理 '-', 'None', '' 等无效值"""
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            value = value.strip()
            # 处理无效值
            if value in ('-', 'None', 'N/A', '', 'null'):
                return default
            try:
                return float(value)
            except ValueError:
                return default
        return default
    
    async def _get_stock_info_alpha_vantage(self, symbol: str, market: str) -> Optional[StockInfo]:
        """从Alpha Vantage获取股票信息（备用）"""
        if settings.alpha_vantage_api_key == "demo":
            return None
        
        try:
            # 获取公司概览
            params = {
                "function": "OVERVIEW",
                "symbol": symbol.replace('.HK', ''),  # Alpha Vantage不支持.HK后缀
                "apikey": settings.alpha_vantage_api_key
            }
            
            response = await self.http_client.get(self.alpha_vantage_base, params=params)
            data = response.json()
            
            if not data or 'Symbol' not in data:
                return None
            
            return StockInfo(
                symbol=symbol,
                name=data.get('Name', ''),
                market=market,
                sector=data.get('Sector', ''),
                industry=data.get('Industry', ''),
                market_cap=self._safe_float(data.get('MarketCapitalization')),
                pe_ratio=self._safe_float(data.get('PERatio')),
                pb_ratio=self._safe_float(data.get('PriceToBookRatio')),
                dividend_yield=self._safe_float(data.get('DividendYield')),
                description=data.get('Description', '')[:500] if data.get('Description') else ''
            )
        except Exception as e:
            logger.error(f"Alpha Vantage获取股票信息失败: {symbol}, 错误: {e}")
            return None
    
    # ==================== 价格数据 ====================
    
    async def get_current_price(self, symbol: str) -> Optional[MarketOverview]:
        """
        获取当前价格概览（带重试和缓存）
        
        Args:
            symbol: 股票代码
        """
        std_symbol, market = parse_symbol(symbol)
        
        # 尝试yfinance（带重试）
        overview = await self._get_current_price_yfinance_with_retry(std_symbol)
        
        # 如果完全失败，最后尝试Alpha Vantage
        if not overview:
            overview = await self._get_current_price_alpha_vantage(std_symbol)
        
        return overview
    
    async def _get_current_price_yfinance_with_retry(self, symbol: str) -> Optional[MarketOverview]:
        """
        从yfinance获取当前价格（使用fast_info，更轻量）
        参考: https://ranaroussi.github.io/yfinance/reference/index.html
        """
        for attempt in range(self._retry_count):
            await self._rate_limit()
            
            try:
                ticker = yf.Ticker(symbol)
                # 使用 fast_info 替代 info，减少 API 调用
                fast = ticker.fast_info
                
                if not fast:
                    return None
                
                return MarketOverview(
                    symbol=symbol,
                    current_price=fast.get('lastPrice', 0) or fast.get('regularMarketPrice', 0),
                    prev_close=fast.get('previousClose', 0) or fast.get('regularMarketPreviousClose', 0),
                    open_price=fast.get('open', 0) or fast.get('regularMarketOpen', 0),
                    day_high=fast.get('dayHigh', 0) or fast.get('regularMarketDayHigh', 0),
                    day_low=fast.get('dayLow', 0) or fast.get('regularMarketDayLow', 0),
                    volume=fast.get('lastVolume', 0) or fast.get('regularMarketVolume', 0),
                    avg_volume=fast.get('threeMonthAverageVolume'),
                    week_52_high=fast.get('fiftyTwoWeekHigh'),
                    week_52_low=fast.get('fiftyTwoWeekLow')
                )
            except Exception as e:
                error_msg = str(e)
                if 'RateLimit' in error_msg or 'Too Many Requests' in error_msg:
                    if attempt < self._retry_count - 1:
                        wait_time = self._retry_delay * (attempt + 1)
                        logger.warning(f"yfinance限速，{wait_time}秒后重试 ({attempt + 1}/{self._retry_count}): {symbol}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"yfinance限速，已达最大重试次数: {symbol}")
                else:
                    logger.error(f"yfinance获取当前价格失败: {symbol}, 错误: {e}")
                    break
        
        return None
    
    async def _get_current_price_alpha_vantage(self, symbol: str) -> Optional[MarketOverview]:
        """从Alpha Vantage获取当前价格（备用）"""
        quote = await self.get_alpha_vantage_quote(symbol)
        
        if not quote:
            return None
        
        return MarketOverview(
            symbol=symbol,
            current_price=quote.get('price', 0),
            prev_close=quote.get('price', 0) - quote.get('change', 0),
            open_price=quote.get('price', 0),
            day_high=quote.get('price', 0),
            day_low=quote.get('price', 0),
            volume=quote.get('volume', 0)
        )
    
    async def get_price_history(
        self, 
        symbol: str, 
        period: str = "1mo",
        interval: str = "1d"
    ) -> List[StockPrice]:
        """
        获取历史价格数据（优化版）
        
        Args:
            symbol: 股票代码
            period: 时间周期 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: K线间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
        """
        std_symbol, market = parse_symbol(symbol)
        
        # 周期对应的交易日数
        period_to_days = {
            "1mo": 22,
            "3mo": 65,
            "6mo": 130,
            "1y": 252,
            "2y": 504,
        }
        target_days = period_to_days.get(period, 65)
        
        # 检查缓存，如果有且数据足够，直接使用
        cached = self._get_from_cache(std_symbol)
        if cached and len(cached) >= target_days:
            cache_time = self._price_cache.get(std_symbol, {}).get('timestamp')
            if cache_time and (datetime.now() - cache_time).seconds < self._cache_ttl:
                logger.info(f"使用新鲜缓存数据: {std_symbol}, 截取 {target_days} 条")
                return cached[-target_days:]
        
        # 直接使用用户请求的周期（不再总是获取1年）
        # 这样减少数据量，降低被限速概率
        prices = await self._get_price_history_yfinance(std_symbol, period, interval)
        
        # yfinance 失败时的处理
        if not prices:
            # 优先使用缓存（即使不够长）
            if cached:
                logger.info(f"yfinance失败，使用缓存数据: {std_symbol}, 共 {len(cached)} 条")
                prices = cached[-target_days:] if len(cached) > target_days else cached
            else:
                # 最后尝试 Alpha Vantage（只请求100条以内）
                logger.info(f"尝试 Alpha Vantage 作为最后备用: {std_symbol}")
                prices = await self._get_price_history_alpha_vantage(std_symbol, period)
        
        # 缓存成功获取的数据
        if prices and len(prices) > 0:
            self._save_to_cache(std_symbol, prices)
        
        # 根据请求的周期截取数据
        if prices and len(prices) > target_days:
            prices = prices[-target_days:]
            logger.info(f"根据周期 {period} 截取数据: {len(prices)} 条")
        
        return prices
    
    def _get_from_cache(self, symbol: str) -> Optional[List[StockPrice]]:
        """从缓存获取价格数据"""
        if symbol not in self._price_cache:
            return None
        
        cached = self._price_cache[symbol]
        cache_time = cached.get('timestamp')
        
        # 检查缓存是否过期
        if cache_time and (datetime.now() - cache_time).seconds < self._cache_ttl:
            return cached.get('prices', [])
        
        return cached.get('prices', [])  # 即使过期也返回（总比没有好）
    
    def _save_to_cache(self, symbol: str, prices: List[StockPrice]):
        """保存价格数据到缓存"""
        self._price_cache[symbol] = {
            'prices': prices,
            'timestamp': datetime.now()
        }
    
    async def _get_price_history_yfinance(self, symbol: str, period: str, interval: str) -> List[StockPrice]:
        """
        从yfinance获取历史价格
        参考知乎教程: https://zhuanlan.zhihu.com/p/674945970
        使用 start/end 日期范围获取数据，更精确且稳定
        """
        # 将 period 转换为日期范围
        period_to_days = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
        }
        days = period_to_days.get(period, 90)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        for attempt in range(self._retry_count):
            await self._rate_limit()
            
            try:
                # 使用 start/end 日期范围，比 period 更稳定
                # 参考: ticker.history(start='2023-01-01', end=end_date)
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True  # 自动调整股息和拆分
                )
                
                if df.empty:
                    logger.warning(f"yfinance无历史数据: {symbol}")
                    return []
                
                prices = []
                prev_close = None
                
                for idx, row in df.iterrows():
                    close_val = float(row['Close'])
                    open_val = float(row['Open'])
                    high_val = float(row['High'])
                    low_val = float(row['Low'])
                    volume_val = int(row['Volume'])
                    
                    price = StockPrice(
                        symbol=symbol,
                        timestamp=idx.to_pydatetime(),
                        open=open_val,
                        high=high_val,
                        low=low_val,
                        close=close_val,
                        volume=volume_val,
                        adj_close=close_val
                    )
                    
                    if prev_close:
                        price.calculate_change(prev_close)
                    
                    prev_close = close_val
                    prices.append(price)
                
                logger.info(f"yfinance获取 {symbol} 历史数据: {len(prices)} 条 ({start_date} ~ {end_date})")
                
                # 保存到本地缓存文件（减少后续API调用）
                self._save_to_csv_cache(symbol, df)
                
                return prices
                
            except Exception as e:
                error_msg = str(e)
                if 'RateLimit' in error_msg or 'Too Many Requests' in error_msg:
                    if attempt < self._retry_count - 1:
                        wait_time = self._retry_delay * (attempt + 1)
                        logger.warning(f"yfinance限速，{wait_time}秒后重试 ({attempt + 1}/{self._retry_count}): {symbol}")
                        await asyncio.sleep(wait_time)
                    else:
                        # 尝试从本地CSV缓存读取
                        cached_prices = self._load_from_csv_cache(symbol, days)
                        if cached_prices:
                            logger.info(f"从本地缓存读取 {symbol}: {len(cached_prices)} 条")
                            return cached_prices
                        logger.warning(f"yfinance限速，已达最大重试次数: {symbol}")
                else:
                    logger.error(f"yfinance获取历史价格失败: {symbol}, 错误: {e}")
                    break
        
        return []
    
    def _save_to_csv_cache(self, symbol: str, df):
        """保存数据到本地CSV缓存"""
        try:
            import os
            cache_dir = "data/cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = f"{cache_dir}/{symbol}_history.csv"
            df.to_csv(cache_file)
            logger.debug(f"保存 {symbol} 数据到本地缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _load_from_csv_cache(self, symbol: str, days: int) -> List[StockPrice]:
        """从本地CSV缓存读取数据"""
        try:
            import os
            import pandas as pd
            cache_file = f"data/cache/{symbol}_history.csv"
            
            if not os.path.exists(cache_file):
                return []
            
            # 检查文件修改时间（超过1天则不使用）
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_time).days > 1:
                return []
            
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # 只取最近需要的天数
            if len(df) > days:
                df = df.tail(days)
            
            prices = []
            prev_close = None
            
            for idx, row in df.iterrows():
                price = StockPrice(
                    symbol=symbol,
                    timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adj_close=float(row['Close'])
                )
                if prev_close:
                    price.calculate_change(prev_close)
                prev_close = float(row['Close'])
                prices.append(price)
            
            return prices
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            return []
    
    async def _get_price_history_alpha_vantage(self, symbol: str, period: str = "3mo") -> List[StockPrice]:
        """从Alpha Vantage获取历史价格（备用，免费版限制较多）"""
        if settings.alpha_vantage_api_key == "demo":
            logger.warning("Alpha Vantage API key 未配置，跳过")
            return []
        
        # 根据周期确定需要的数据量
        period_to_days = {
            "1mo": 22,
            "3mo": 65,
            "6mo": 130,
            "1y": 252,
            "2y": 504,
        }
        target_days = min(period_to_days.get(period, 65), 100)  # 免费版最多100条
        
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol.replace('.HK', ''),
                "outputsize": "compact",  # 免费版只能用 compact（最近100条）
                "apikey": settings.alpha_vantage_api_key
            }
            
            response = await self.http_client.get(self.alpha_vantage_base, params=params)
            data = response.json()
            
            # 检查API错误响应
            if "Error Message" in data:
                logger.warning(f"Alpha Vantage API错误: {data['Error Message']}")
                return []
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage API限制: {data['Note']}")
                return []
            
            if "Information" in data:
                logger.warning(f"Alpha Vantage API信息: {data['Information']}")
                return []
            
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                logger.warning(f"Alpha Vantage无历史数据: {symbol}, 响应: {list(data.keys())}")
                return []
            
            prices = []
            prev_close = None
            
            # 按日期排序并只取需要的数量
            sorted_dates = sorted(time_series.items())
            # 只取最近的 target_days 条数据
            sorted_dates = sorted_dates[-target_days:] if len(sorted_dates) > target_days else sorted_dates
            
            for date_str, values in sorted_dates:
                price = StockPrice(
                    symbol=symbol,
                    timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    volume=int(values["5. volume"])
                )
                
                if prev_close:
                    price.calculate_change(prev_close)
                
                prev_close = price.close
                prices.append(price)
            
            logger.info(f"Alpha Vantage获取 {symbol} 历史数据: {len(prices)} 条 (周期: {period})")
            return prices
            
        except Exception as e:
            logger.error(f"Alpha Vantage获取历史价格失败: {symbol}, 错误: {e}")
            return []
    
    async def get_intraday_prices(self, symbol: str, interval: str = "5m") -> List[StockPrice]:
        """
        获取盘中分时数据
        
        Args:
            symbol: 股票代码
            interval: 时间间隔 (1m, 5m, 15m, 30m, 1h)
        """
        return await self.get_price_history(symbol, period="1d", interval=interval)
    
    # ==================== 新闻数据 ====================
    
    async def get_news(self, symbol: str, limit: int = 20) -> List[StockNews]:
        """
        获取股票相关新闻
        
        优先使用Finnhub，备用yfinance
        """
        std_symbol, market = parse_symbol(symbol)
        news_list = []
        
        # 尝试Finnhub
        if self.finnhub_client:
            try:
                news_list = await self._get_finnhub_news(std_symbol, limit)
            except Exception as e:
                logger.warning(f"Finnhub新闻获取失败: {e}")
        
        # 如果Finnhub失败或无数据，使用yfinance
        if not news_list:
            try:
                news_list = await self._get_yfinance_news(std_symbol, limit)
            except Exception as e:
                logger.warning(f"yfinance新闻获取失败: {e}")
        
        return news_list[:limit]
    
    async def _get_finnhub_news(self, symbol: str, limit: int) -> List[StockNews]:
        """从Finnhub获取新闻"""
        if not self.finnhub_client:
            return []
        
        # Finnhub需要使用不带.HK的代码
        clean_symbol = symbol.replace('.HK', '')
        
        # 获取最近7天的新闻
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        
        try:
            # Finnhub是同步的，用线程池执行
            loop = asyncio.get_event_loop()
            news_data = await loop.run_in_executor(
                None,
                lambda: self.finnhub_client.company_news(
                    clean_symbol,
                    _from=from_date.strftime('%Y-%m-%d'),
                    to=to_date.strftime('%Y-%m-%d')
                )
            )
            
            news_list = []
            for item in news_data[:limit]:
                news = StockNews(
                    symbol=symbol,
                    title=item.get('headline', ''),
                    summary=item.get('summary', ''),
                    source=item.get('source', ''),
                    url=item.get('url', ''),
                    published_at=datetime.fromtimestamp(item.get('datetime', 0)),
                    sentiment=self._parse_finnhub_sentiment(item.get('sentiment')),
                    relevance=item.get('relevance')
                )
                news_list.append(news)
            
            logger.info(f"从Finnhub获取 {symbol} 新闻: {len(news_list)} 条")
            return news_list
            
        except Exception as e:
            logger.error(f"Finnhub新闻获取失败: {e}")
            return []
    
    async def _get_yfinance_news(self, symbol: str, limit: int) -> List[StockNews]:
        """从yfinance获取新闻（带重试机制）"""
        for attempt in range(self._retry_count):
            await self._rate_limit()  # 请求限速
            
            try:
                ticker = yf.Ticker(symbol)
                news_data = ticker.news
                
                if not news_data:
                    return []
                
                news_list = []
                for item in news_data[:limit]:
                    published_at = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                    
                    news = StockNews(
                        symbol=symbol,
                        title=item.get('title', ''),
                        summary='',  # yfinance不提供摘要
                        source=item.get('publisher', ''),
                        url=item.get('link', ''),
                        published_at=published_at
                    )
                    news_list.append(news)
                
                logger.info(f"从yfinance获取 {symbol} 新闻: {len(news_list)} 条")
                return news_list
                
            except Exception as e:
                error_msg = str(e)
                if 'RateLimit' in error_msg or 'Too Many Requests' in error_msg:
                    if attempt < self._retry_count - 1:
                        wait_time = self._retry_delay * (attempt + 1)
                        logger.warning(f"yfinance新闻限速，{wait_time}秒后重试 ({attempt + 1}/{self._retry_count})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"yfinance新闻获取失败: {e}")
                else:
                    logger.error(f"yfinance新闻获取失败: {e}")
                    break
        
        return []
    
    def _parse_finnhub_sentiment(self, sentiment_data: Optional[Dict]) -> Optional[str]:
        """解析Finnhub情绪数据"""
        if not sentiment_data:
            return None
        
        score = sentiment_data.get('bullishPercent', 50)
        if score > 60:
            return "positive"
        elif score < 40:
            return "negative"
        return "neutral"
    
    # ==================== Alpha Vantage备用 ====================
    
    async def get_alpha_vantage_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        从Alpha Vantage获取实时报价（备用）
        
        注意：免费版每分钟5次请求限制
        """
        if settings.alpha_vantage_api_key == "demo":
            logger.warning("Alpha Vantage使用demo key，功能受限")
        
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": settings.alpha_vantage_api_key
            }
            
            response = await self.http_client.get(self.alpha_vantage_base, params=params)
            data = response.json()
            
            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    "symbol": quote.get("01. symbol"),
                    "price": float(quote.get("05. price", 0)),
                    "change": float(quote.get("09. change", 0)),
                    "change_percent": quote.get("10. change percent", "0%"),
                    "volume": int(quote.get("06. volume", 0))
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Alpha Vantage请求失败: {e}")
            return None
    
    # ==================== 财报数据 ====================
    
    async def get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        """
        获取财报数据
        参考知乎教程: https://zhuanlan.zhihu.com/p/674945970
        使用 get_earnings_dates() 替代废弃的 earnings
        
        Args:
            symbol: 股票代码
            
        Returns:
            财报数据列表
        """
        std_symbol, _ = parse_symbol(symbol)
        earnings_list = []
        
        await self._rate_limit()
        
        # 从yfinance获取财报
        try:
            ticker = yf.Ticker(std_symbol)
            
            # 使用 earnings_dates 替代废弃的 earnings
            # 参考: ticker.get_earnings_dates()
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    for idx, row in earnings_dates.head(12).iterrows():  # 最近12个季度
                        earnings_list.append({
                            "date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                            "type": "quarterly",
                            "eps_estimate": row.get('EPS Estimate'),
                            "reported_eps": row.get('Reported EPS'),
                            "surprise": row.get('Surprise(%)'),
                            "source": "yfinance"
                        })
            except Exception:
                pass
            
            # 获取收益历史 earnings_history
            try:
                earnings_hist = ticker.earnings_history
                if earnings_hist is not None and not earnings_hist.empty:
                    for idx, row in earnings_hist.iterrows():
                        earnings_list.append({
                            "date": str(idx),
                            "type": "history",
                            "eps_actual": row.get('epsActual'),
                            "eps_estimate": row.get('epsEstimate'),
                            "eps_diff": row.get('epsDifference'),
                            "surprise_percent": row.get('surprisePercent'),
                            "source": "yfinance"
                        })
            except Exception:
                pass
                    
            logger.info(f"获取 {std_symbol} 财报数据: {len(earnings_list)} 条")
            
        except Exception as e:
            logger.warning(f"yfinance获取财报失败: {std_symbol}, 错误: {e}")
        
        # 从Finnhub获取财报日历
        if self.finnhub_client:
            try:
                loop = asyncio.get_event_loop()
                earnings_calendar = await loop.run_in_executor(
                    None,
                    lambda: self.finnhub_client.earnings_calendar(
                        symbol=std_symbol.replace('.HK', ''),
                        _from=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        to=datetime.now().strftime('%Y-%m-%d')
                    )
                )
                
                if earnings_calendar and 'earningsCalendar' in earnings_calendar:
                    for item in earnings_calendar['earningsCalendar']:
                        earnings_list.append({
                            "date": item.get('date', ''),
                            "type": "calendar",
                            "eps_actual": item.get('epsActual'),
                            "eps_estimate": item.get('epsEstimate'),
                            "revenue_actual": item.get('revenueActual'),
                            "revenue_estimate": item.get('revenueEstimate'),
                            "source": "finnhub"
                        })
                        
            except Exception as e:
                logger.warning(f"Finnhub获取财报日历失败: {e}")
        
        return earnings_list
    
    async def get_financials(self, symbol: str) -> Dict[str, Any]:
        """
        获取完整财务数据
        参考知乎教程: https://zhuanlan.zhihu.com/p/674945970
        包括：资产负债表、现金流、收入报表、分析预测
        """
        std_symbol, _ = parse_symbol(symbol)
        financials = {
            "balance_sheet": [],
            "cashflow": [],
            "income_stmt": [],
            "analysis": {},
            "institutional_holders": [],
            "major_holders": None
        }
        
        await self._rate_limit()
        
        try:
            ticker = yf.Ticker(std_symbol)
            
            # 1. 资产负债表 get_balance_sheet()
            try:
                bs = ticker.balance_sheet
                if bs is not None and not bs.empty:
                    for col in bs.columns[:4]:  # 最近4个季度
                        financials["balance_sheet"].append({
                            "date": str(col.date()) if hasattr(col, 'date') else str(col),
                            "total_assets": self._safe_float(bs.loc['Total Assets', col]) if 'Total Assets' in bs.index else None,
                            "total_liabilities": self._safe_float(bs.loc['Total Liabilities Net Minority Interest', col]) if 'Total Liabilities Net Minority Interest' in bs.index else None,
                            "total_equity": self._safe_float(bs.loc['Total Equity Gross Minority Interest', col]) if 'Total Equity Gross Minority Interest' in bs.index else None,
                            "cash": self._safe_float(bs.loc['Cash And Cash Equivalents', col]) if 'Cash And Cash Equivalents' in bs.index else None,
                        })
                    logger.info(f"获取 {std_symbol} 资产负债表: {len(financials['balance_sheet'])} 条")
            except Exception as e:
                logger.debug(f"资产负债表获取失败: {e}")
            
            # 2. 现金流 get_cashflow()
            try:
                cf = ticker.cashflow
                if cf is not None and not cf.empty:
                    for col in cf.columns[:4]:
                        financials["cashflow"].append({
                            "date": str(col.date()) if hasattr(col, 'date') else str(col),
                            "operating_cashflow": self._safe_float(cf.loc['Operating Cash Flow', col]) if 'Operating Cash Flow' in cf.index else None,
                            "investing_cashflow": self._safe_float(cf.loc['Investing Cash Flow', col]) if 'Investing Cash Flow' in cf.index else None,
                            "financing_cashflow": self._safe_float(cf.loc['Financing Cash Flow', col]) if 'Financing Cash Flow' in cf.index else None,
                            "free_cashflow": self._safe_float(cf.loc['Free Cash Flow', col]) if 'Free Cash Flow' in cf.index else None,
                        })
                    logger.info(f"获取 {std_symbol} 现金流: {len(financials['cashflow'])} 条")
            except Exception as e:
                logger.debug(f"现金流获取失败: {e}")
            
            # 3. 收入报表 income_stmt
            try:
                income = ticker.income_stmt
                if income is not None and not income.empty:
                    for col in income.columns[:4]:
                        financials["income_stmt"].append({
                            "date": str(col.date()) if hasattr(col, 'date') else str(col),
                            "total_revenue": self._safe_float(income.loc['Total Revenue', col]) if 'Total Revenue' in income.index else None,
                            "gross_profit": self._safe_float(income.loc['Gross Profit', col]) if 'Gross Profit' in income.index else None,
                            "operating_income": self._safe_float(income.loc['Operating Income', col]) if 'Operating Income' in income.index else None,
                            "net_income": self._safe_float(income.loc['Net Income', col]) if 'Net Income' in income.index else None,
                        })
                    logger.info(f"获取 {std_symbol} 收入报表: {len(financials['income_stmt'])} 条")
            except Exception as e:
                logger.debug(f"收入报表获取失败: {e}")
            
            # 4. 分析预测 get_analysis() 相关
            try:
                # 增长估计
                growth = ticker.growth_estimates
                if growth is not None and not growth.empty and std_symbol in growth.columns:
                    financials["analysis"]["growth_estimates"] = {
                        "current_qtr": self._safe_float(growth.loc['0q', std_symbol]) if '0q' in growth.index else None,
                        "next_qtr": self._safe_float(growth.loc['+1q', std_symbol]) if '+1q' in growth.index else None,
                        "current_year": self._safe_float(growth.loc['0y', std_symbol]) if '0y' in growth.index else None,
                        "next_year": self._safe_float(growth.loc['+1y', std_symbol]) if '+1y' in growth.index else None,
                    }
                
                # 收益估计
                earnings_est = ticker.earnings_estimate
                if earnings_est is not None and not earnings_est.empty:
                    financials["analysis"]["earnings_estimate"] = earnings_est.to_dict()
                
                # 收入估计
                revenue_est = ticker.revenue_estimate
                if revenue_est is not None and not revenue_est.empty:
                    financials["analysis"]["revenue_estimate"] = revenue_est.to_dict()
                    
                # 目标价
                target = ticker.analyst_price_targets
                if target:
                    financials["analysis"]["price_targets"] = {
                        "current": target.get('current'),
                        "low": target.get('low'),
                        "high": target.get('high'),
                        "mean": target.get('mean'),
                        "median": target.get('median'),
                    }
                    
                logger.info(f"获取 {std_symbol} 分析预测数据")
            except Exception as e:
                logger.debug(f"分析预测获取失败: {e}")
            
            # 5. 机构持有者 get_institutional_holders()
            try:
                inst = ticker.institutional_holders
                if inst is not None and not inst.empty:
                    for _, row in inst.head(10).iterrows():
                        financials["institutional_holders"].append({
                            "holder": row.get('Holder', ''),
                            "shares": row.get('Shares', 0),
                            "date_reported": str(row.get('Date Reported', '')),
                            "percent_out": row.get('% Out', 0),
                            "value": row.get('Value', 0),
                        })
                    logger.info(f"获取 {std_symbol} 机构持有者: {len(financials['institutional_holders'])} 条")
            except Exception as e:
                logger.debug(f"机构持有者获取失败: {e}")
            
            # 6. 主要持有者
            try:
                major = ticker.major_holders
                if major is not None and not major.empty:
                    financials["major_holders"] = major.to_dict()
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"获取财务数据失败: {std_symbol}, 错误: {e}")
        
        return financials
    
    async def get_analyst_ratings(self, symbol: str) -> List[Dict[str, Any]]:
        """
        获取机构评级和目标价
        
        Args:
            symbol: 股票代码
            
        Returns:
            机构评级列表
        """
        std_symbol, _ = parse_symbol(symbol)
        ratings_list = []
        
        # 从yfinance获取推荐
        try:
            ticker = yf.Ticker(std_symbol)
            recommendations = ticker.recommendations
            
            if recommendations is not None and not recommendations.empty:
                for idx, row in recommendations.tail(20).iterrows():
                    ratings_list.append({
                        "date": str(idx),
                        "firm": row.get('Firm', ''),
                        "to_grade": row.get('To Grade', ''),
                        "from_grade": row.get('From Grade', ''),
                        "action": row.get('Action', ''),
                        "source": "yfinance"
                    })
                    
            logger.info(f"获取 {std_symbol} 机构评级: {len(ratings_list)} 条")
            
        except Exception as e:
            logger.warning(f"yfinance获取机构评级失败: {std_symbol}, 错误: {e}")
        
        # 从Finnhub获取评级
        if self.finnhub_client:
            try:
                loop = asyncio.get_event_loop()
                
                # 获取推荐趋势
                rec_trends = await loop.run_in_executor(
                    None,
                    lambda: self.finnhub_client.recommendation_trends(std_symbol.replace('.HK', ''))
                )
                
                if rec_trends:
                    for item in rec_trends[:10]:
                        ratings_list.append({
                            "date": item.get('period', ''),
                            "strong_buy": item.get('strongBuy', 0),
                            "buy": item.get('buy', 0),
                            "hold": item.get('hold', 0),
                            "sell": item.get('sell', 0),
                            "strong_sell": item.get('strongSell', 0),
                            "source": "finnhub_trend"
                        })
                
                # 获取目标价
                price_target = await loop.run_in_executor(
                    None,
                    lambda: self.finnhub_client.price_target(std_symbol.replace('.HK', ''))
                )
                
                if price_target:
                    ratings_list.append({
                        "date": datetime.now().strftime('%Y-%m-%d'),
                        "target_high": price_target.get('targetHigh'),
                        "target_low": price_target.get('targetLow'),
                        "target_mean": price_target.get('targetMean'),
                        "target_median": price_target.get('targetMedian'),
                        "source": "finnhub_target"
                    })
                    
            except Exception as e:
                # Finnhub 免费版不支持某些评级功能，这是正常的
                if '403' in str(e) or 'access' in str(e).lower():
                    logger.debug(f"Finnhub免费版无权访问评级数据（正常）: {std_symbol}")
                else:
                    logger.warning(f"Finnhub获取评级失败: {e}")
        
        return ratings_list
    
    async def get_insider_transactions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        获取内部交易数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            内部交易列表
        """
        std_symbol, _ = parse_symbol(symbol)
        transactions = []
        
        # 从yfinance获取内部交易
        try:
            ticker = yf.Ticker(std_symbol)
            insider = ticker.insider_transactions
            
            if insider is not None and not insider.empty:
                for idx, row in insider.head(20).iterrows():
                    transactions.append({
                        "date": str(row.get('Start Date', '')),
                        "insider": row.get('Insider', ''),
                        "position": row.get('Position', ''),
                        "transaction": row.get('Transaction', ''),
                        "shares": row.get('Shares', 0),
                        "value": row.get('Value', 0),
                        "source": "yfinance"
                    })
                    
            logger.info(f"获取 {std_symbol} 内部交易: {len(transactions)} 条")
            
        except Exception as e:
            logger.warning(f"获取内部交易失败: {std_symbol}, 错误: {e}")
        
        # 从Finnhub获取内部交易
        if self.finnhub_client:
            try:
                loop = asyncio.get_event_loop()
                insider_data = await loop.run_in_executor(
                    None,
                    lambda: self.finnhub_client.stock_insider_transactions(
                        std_symbol.replace('.HK', ''),
                        _from=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
                        to=datetime.now().strftime('%Y-%m-%d')
                    )
                )
                
                if insider_data and 'data' in insider_data:
                    for item in insider_data['data'][:20]:
                        transactions.append({
                            "date": item.get('transactionDate', ''),
                            "insider": item.get('name', ''),
                            "shares": item.get('share', 0),
                            "change": item.get('change', 0),
                            "transaction_code": item.get('transactionCode', ''),
                            "source": "finnhub"
                        })
                        
            except Exception as e:
                logger.warning(f"Finnhub获取内部交易失败: {e}")
        
        return transactions
    
    async def get_sec_filings(self, symbol: str) -> List[Dict[str, Any]]:
        """
        获取SEC文件/公告
        
        Args:
            symbol: 股票代码
            
        Returns:
            SEC文件列表
        """
        std_symbol, _ = parse_symbol(symbol)
        filings = []
        
        # 从Finnhub获取SEC文件
        if self.finnhub_client:
            try:
                loop = asyncio.get_event_loop()
                sec_data = await loop.run_in_executor(
                    None,
                    lambda: self.finnhub_client.filings(
                        symbol=std_symbol.replace('.HK', ''),
                        _from=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        to=datetime.now().strftime('%Y-%m-%d')
                    )
                )
                
                if sec_data:
                    for item in sec_data[:30]:
                        filings.append({
                            "date": item.get('filedDate', ''),
                            "form": item.get('form', ''),
                            "access_number": item.get('accessNumber', ''),
                            "url": item.get('reportUrl', ''),
                            "source": "finnhub"
                        })
                        
                logger.info(f"获取 {std_symbol} SEC文件: {len(filings)} 条")
                        
            except Exception as e:
                logger.warning(f"Finnhub获取SEC文件失败: {e}")
        
        return filings
    
    async def get_company_news_extended(self, symbol: str, limit: int = 50) -> List[StockNews]:
        """
        获取扩展的公司新闻（多数据源，至少50条）
        
        Args:
            symbol: 股票代码
            limit: 目标新闻数量
            
        Returns:
            新闻列表
        """
        std_symbol, _ = parse_symbol(symbol)
        all_news = []
        
        # 从Finnhub获取新闻（主要来源）
        if self.finnhub_client:
            try:
                finnhub_news = await self._get_finnhub_news(std_symbol, limit=30)
                all_news.extend(finnhub_news)
            except Exception as e:
                logger.warning(f"Finnhub新闻获取失败: {e}")
        
        # 从yfinance获取新闻
        try:
            yf_news = await self._get_yfinance_news(std_symbol, limit=20)
            all_news.extend(yf_news)
        except Exception as e:
            logger.warning(f"yfinance新闻获取失败: {e}")
        
        # 去重（基于标题）
        seen_titles = set()
        unique_news = []
        for news in all_news:
            title_key = news.title[:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)
        
        # 按时间排序
        unique_news.sort(key=lambda x: x.published_at, reverse=True)
        
        logger.info(f"获取 {std_symbol} 扩展新闻: {len(unique_news)} 条")
        return unique_news[:limit]
    
    # ==================== 批量获取 ====================
    
    async def fetch_all_data(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """
        一次性获取股票所有数据（优化版）
        
        关键优化：
        1. 复用单一 Ticker 对象，减少创建开销
        2. 顺序获取而非并发，避免限速
        3. 优先使用缓存数据
        
        Args:
            symbol: 股票代码
            period: 历史数据周期
        
        Returns:
            包含完整数据的字典
        """
        std_symbol, market = parse_symbol(symbol)
        
        # 初始化结果
        result = {
            "symbol": std_symbol,
            "market": market,
            "period": period,
            "info": None,
            "current_price": None,
            "price_history": [],
            "news": [],
            "earnings": [],
            "analyst_ratings": [],
            "insider_transactions": [],
            "sec_filings": [],
            "financials": {},
            "fetched_at": datetime.now().isoformat()
        }
        
        # 获取或创建 Ticker 对象（复用）
        ticker = self._get_ticker(std_symbol)
        
        # ========== 1. 从单一 Ticker 对象获取所有数据 ==========
        # 这样只需要一次网络请求获取基础数据
        await self._rate_limit()
        
        try:
            # 一次性获取多个属性（yfinance 内部会缓存）
            logger.info(f"开始获取 {std_symbol} 数据...")
            
            # 基本信息（使用 fast_info 更快）
            try:
                fast_info = ticker.fast_info
                if fast_info:
                    result["current_price"] = MarketOverview(
                        symbol=std_symbol,
                        current_price=fast_info.get('lastPrice', 0) or fast_info.get('regularMarketPrice', 0),
                        prev_close=fast_info.get('previousClose', 0),
                        open_price=fast_info.get('open', 0),
                        day_high=fast_info.get('dayHigh', 0),
                        day_low=fast_info.get('dayLow', 0),
                        volume=fast_info.get('lastVolume', 0),
                        avg_volume=fast_info.get('threeMonthAverageVolume'),
                        week_52_high=fast_info.get('fiftyTwoWeekHigh'),
                        week_52_low=fast_info.get('fiftyTwoWeekLow')
                    )
                    logger.info(f"获取 {std_symbol} 价格信息成功")
            except Exception as e:
                if 'Too Many Requests' in str(e):
                    self._mark_rate_limited(60)
                logger.warning(f"获取价格信息失败: {e}")
            
            await asyncio.sleep(0.5)  # 小间隔
            
            # 股票详细信息（从缓存或 info）
            cached_info = self._get_info_from_cache(std_symbol)
            if cached_info:
                result["info"] = cached_info
            else:
                try:
                    info = ticker.info
                    if info and 'symbol' in info:
                        result["info"] = StockInfo(
                            symbol=std_symbol,
                            name=info.get('longName') or info.get('shortName', ''),
                            market=market,
                            sector=info.get('sector', ''),
                            industry=info.get('industry', ''),
                            market_cap=info.get('marketCap'),
                            pe_ratio=info.get('trailingPE'),
                            pb_ratio=info.get('priceToBook'),
                            dividend_yield=info.get('dividendYield'),
                            description=info.get('longBusinessSummary', '')[:500] if info.get('longBusinessSummary') else ''
                        )
                        self._save_info_to_cache(std_symbol, result["info"])
                except Exception as e:
                    if 'Too Many Requests' in str(e):
                        self._mark_rate_limited(60)
                    logger.warning(f"获取详细信息失败: {e}")
            
            await asyncio.sleep(0.5)
            
            # 历史数据
            result["price_history"] = await self.get_price_history(std_symbol, period=period)
            
        except Exception as e:
            logger.error(f"Ticker数据获取失败: {e}")
        
        # ========== 2. 获取新闻（Finnhub 优先，不易限速）==========
        try:
            result["news"] = await self.get_company_news_extended(std_symbol, limit=50)
        except Exception as e:
            logger.warning(f"新闻获取失败: {e}")
        
        # ========== 3. 获取财务数据（如果未被限速）==========
        if not self._rate_limited:
            try:
                result["financials"] = await self.get_financials(std_symbol)
                result["earnings"] = await self.get_earnings(std_symbol)
            except Exception as e:
                logger.warning(f"财务数据获取失败: {e}")
        
        # ========== 4. 获取评级和SEC文件 ==========
        try:
            result["analyst_ratings"] = await self.get_analyst_ratings(std_symbol)
            result["sec_filings"] = await self.get_sec_filings(std_symbol)
        except Exception as e:
            logger.warning(f"评级数据获取失败: {e}")
        
        # 统计
        total_info = sum([
            len(result['news']),
            len(result['earnings']),
            len(result['analyst_ratings']),
            len(result['sec_filings'])
        ])
        
        fin = result.get('financials', {})
        fin_count = sum([
            len(fin.get('balance_sheet', [])),
            len(fin.get('cashflow', [])),
            len(fin.get('income_stmt', [])),
            len(fin.get('institutional_holders', []))
        ])
        
        logger.info(f"完成 {std_symbol} 数据采集 (K线:{len(result['price_history'])}条, 资讯:{total_info}条, 财务:{fin_count}条)")
        
        return result

    # ==================== v2.0 新增方法：轻量级实时数据 ====================
    
    async def fetch_realtime_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时价格快照（超轻量级，用于实时监控）
        
        仅使用 fast_info，最小化 API 开销
        适合每分钟调用一次
        
        Returns:
            {
                'symbol': str,
                'price': float,
                'change': float,
                'change_percent': float,
                'volume': int,
                'timestamp': datetime
            }
        """
        std_symbol, _ = parse_symbol(symbol)
        
        try:
            await self._rate_limit()
            
            ticker = self._get_ticker(std_symbol)
            fast = ticker.fast_info
            
            if not fast:
                logger.warning(f"无法获取实时快照: {std_symbol}")
                return None
            
            # 获取价格
            price = fast.get('lastPrice') or fast.get('regularMarketPrice', 0)
            prev_close = fast.get('previousClose') or fast.get('regularMarketPreviousClose', 0)
            
            # 计算涨跌
            change = price - prev_close if prev_close else 0
            change_percent = (change / prev_close * 100) if prev_close else 0
            
            return {
                'symbol': std_symbol,
                'price': price,
                'prev_close': prev_close,
                'change': change,
                'change_percent': change_percent,
                'volume': fast.get('lastVolume') or fast.get('regularMarketVolume', 0),
                'day_high': fast.get('dayHigh') or fast.get('regularMarketDayHigh', 0),
                'day_low': fast.get('dayLow') or fast.get('regularMarketDayLow', 0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            error_msg = str(e)
            if 'RateLimit' in error_msg or 'Too Many Requests' in error_msg:
                logger.warning(f"实时快照限速: {std_symbol}")
                self._mark_rate_limited(30)
            else:
                logger.error(f"获取实时快照失败: {std_symbol}, 错误: {e}")
            return None
    
    async def fetch_quick_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取快速报价（比 realtime_snapshot 稍详细）
        
        Returns:
            包含更多市场数据的字典
        """
        std_symbol, market = parse_symbol(symbol)
        
        try:
            await self._rate_limit()
            
            ticker = self._get_ticker(std_symbol)
            fast = ticker.fast_info
            
            if not fast:
                return None
            
            price = fast.get('lastPrice') or fast.get('regularMarketPrice', 0)
            prev_close = fast.get('previousClose') or fast.get('regularMarketPreviousClose', 0)
            change = price - prev_close if prev_close else 0
            change_percent = (change / prev_close * 100) if prev_close else 0
            
            return {
                'symbol': std_symbol,
                'market': market,
                'price': price,
                'prev_close': prev_close,
                'open': fast.get('open') or fast.get('regularMarketOpen', 0),
                'change': change,
                'change_percent': change_percent,
                'volume': fast.get('lastVolume') or fast.get('regularMarketVolume', 0),
                'avg_volume': fast.get('threeMonthAverageVolume', 0),
                'day_high': fast.get('dayHigh') or fast.get('regularMarketDayHigh', 0),
                'day_low': fast.get('dayLow') or fast.get('regularMarketDayLow', 0),
                'week_52_high': fast.get('fiftyTwoWeekHigh'),
                'week_52_low': fast.get('fiftyTwoWeekLow'),
                'market_cap': fast.get('marketCap'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"获取快速报价失败: {std_symbol}, 错误: {e}")
            return None
    
    # ==================== v2.0 新增方法：重量级历史数据 ====================
    
    async def fetch_historical_full(
        self, 
        symbol: str, 
        years: int = 2
    ) -> List[StockPrice]:
        """
        获取完整历史数据（重量级，仅首次或过期时调用）
        
        Args:
            symbol: 股票代码
            years: 历史年数（默认2年）
            
        Returns:
            完整的历史价格列表
        """
        std_symbol, _ = parse_symbol(symbol)
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        logger.info(f"全量拉取 {std_symbol} 历史数据: {start_date.date()} 至 {end_date.date()}")
        
        try:
            await self._rate_limit()
            
            ticker = self._get_ticker(std_symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"全量拉取无数据: {std_symbol}")
                return []
            
            prices = []
            prev_close = None
            
            for idx, row in df.iterrows():
                timestamp = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
                close = row['Close']
                
                change = close - prev_close if prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0
                
                prices.append(StockPrice(
                    symbol=std_symbol,
                    timestamp=timestamp,
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=close,
                    volume=int(row['Volume']),
                    adj_close=close,
                    change=change,
                    change_percent=change_pct
                ))
                
                prev_close = close
            
            logger.info(f"全量拉取 {std_symbol} 完成: {len(prices)} 条 ({prices[0].timestamp.date()} ~ {prices[-1].timestamp.date()})")
            
            # 保存到缓存
            self._save_to_cache(std_symbol, prices)
            
            return prices
            
        except Exception as e:
            logger.error(f"全量拉取历史数据失败: {std_symbol}, 错误: {e}")
            return []
    
    async def fetch_historical_incremental(
        self, 
        symbol: str, 
        since_date: datetime
    ) -> List[StockPrice]:
        """
        获取增量历史数据（从指定日期到今天）
        
        Args:
            symbol: 股票代码
            since_date: 起始日期
            
        Returns:
            增量价格列表
        """
        std_symbol, _ = parse_symbol(symbol)
        
        end_date = datetime.now()
        
        # 如果起始日期已经是今天，无需更新
        if since_date.date() >= end_date.date():
            logger.debug(f"{std_symbol}: 数据已是最新，无需增量更新")
            return []
        
        logger.info(f"增量更新 {std_symbol}: {since_date.date()} 至 {end_date.date()}")
        
        try:
            await self._rate_limit()
            
            ticker = self._get_ticker(std_symbol)
            df = ticker.history(
                start=since_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                auto_adjust=True
            )
            
            if df.empty:
                logger.info(f"增量更新无新数据: {std_symbol}")
                return []
            
            prices = []
            prev_close = None
            
            for idx, row in df.iterrows():
                timestamp = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
                close = row['Close']
                
                change = close - prev_close if prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0
                
                prices.append(StockPrice(
                    symbol=std_symbol,
                    timestamp=timestamp,
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=close,
                    volume=int(row['Volume']),
                    adj_close=close,
                    change=change,
                    change_percent=change_pct
                ))
                
                prev_close = close
            
            logger.info(f"增量更新 {std_symbol} 完成: {len(prices)} 条新数据")
            
            return prices
            
        except Exception as e:
            logger.error(f"增量更新失败: {std_symbol}, 错误: {e}")
            return []
    
    async def fetch_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取基本面数据（日更，包含财务和评级）
        
        这是一个重量级方法，建议每日只调用一次
        """
        std_symbol, market = parse_symbol(symbol)
        
        result = {
            'info': None,
            'financials': {},
            'earnings': [],
            'ratings': []
        }
        
        try:
            # 股票信息
            result['info'] = await self.get_stock_info(std_symbol)
            
            await asyncio.sleep(0.5)
            
            # 财务数据
            if not self._rate_limited:
                result['financials'] = await self.get_financials(std_symbol)
                result['earnings'] = await self.get_earnings(std_symbol)
            
            await asyncio.sleep(0.5)
            
            # 评级
            result['ratings'] = await self.get_analyst_ratings(std_symbol)
            
            logger.info(f"获取 {std_symbol} 基本面数据完成")
            
        except Exception as e:
            logger.error(f"获取基本面数据失败: {std_symbol}, 错误: {e}")
        
        return result


# 创建全局实例的便捷函数
def create_data_fetcher() -> DataFetcher:
    """创建数据采集器实例"""
    return DataFetcher()

