"""
股票分析模块
综合分析股票数据，生成分析报告（增强版）
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from loguru import logger

from app.models.stock import StockInfo, StockPrice, StockNews, MarketOverview
from app.models.analysis import AnalysisRecord, TechnicalIndicators, AIContext
from app.services.data_fetcher import DataFetcher
from app.services.sentiment import SentimentAnalyzer
from app.services.strategy import ComprehensiveAnalyzer, Signal
from app.utils.ai_context import AIContextManager
from app.database.schemas import AnalysisRecordDB


class StockAnalyzer:
    """股票分析器（增强版）"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.data_fetcher = DataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.strategy_analyzer = ComprehensiveAnalyzer()
        self.context_manager = AIContextManager(db_session)
    
    async def analyze(self, symbol: str) -> Optional[AnalysisRecord]:
        """
        执行完整的股票分析
        
        Args:
            symbol: 股票代码
            
        Returns:
            分析记录对象
        """
        logger.info(f"开始分析股票: {symbol}")
        
        # 获取所有数据
        data = await self.data_fetcher.fetch_all_data(symbol)
        
        if not data.get("current_price"):
            logger.warning(f"无法获取 {symbol} 的价格数据")
            return None
        
        current_price: MarketOverview = data["current_price"]
        price_history: List[StockPrice] = data.get("price_history", [])
        news_list: List[StockNews] = data.get("news", [])
        stock_info: Optional[StockInfo] = data.get("info")
        earnings = data.get("earnings", [])
        analyst_ratings = data.get("analyst_ratings", [])
        
        # 计算技术指标
        technical = self._calculate_technical_indicators(price_history)
        
        # 执行策略分析
        strategy_result = self._run_strategy_analysis(price_history)
        
        # 分析新闻情绪
        if news_list:
            news_list = self.sentiment_analyzer.analyze_news_list(news_list)
        
        sentiment_stats = self.sentiment_analyzer.get_overall_sentiment(news_list)
        news_summary = self.sentiment_analyzer.generate_sentiment_summary(news_list)
        
        # 分析机构评级
        ratings_summary = self._analyze_analyst_ratings(analyst_ratings)
        
        # 分析财报
        earnings_summary = self._analyze_earnings(earnings)
        
        # 提取关键事件
        key_events = self._extract_key_events(news_list, earnings, analyst_ratings)
        
        # 生成AI分析摘要
        ai_summary = self._generate_ai_summary(
            symbol=symbol,
            price=current_price,
            technical=technical,
            sentiment=sentiment_stats,
            strategy=strategy_result,
            ratings=ratings_summary,
            earnings=earnings_summary,
            key_events=key_events
        )
        
        # 创建分析记录
        record = AnalysisRecord(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price.current_price,
            price_change=current_price.change,
            price_change_percent=current_price.change_percent,
            volume=current_price.volume,
            technical=technical,
            sentiment_score=sentiment_stats.get("score", 0),
            sentiment_label=sentiment_stats.get("overall", "neutral"),
            news_count=len(news_list),
            news_summary=news_summary,
            ai_summary=ai_summary,
            strategy_signal=strategy_result.get("signal") if strategy_result else None,
            strategy_confidence=strategy_result.get("confidence") if strategy_result else None
        )
        
        # 保存到数据库
        self._save_analysis_record(record)
        
        # 更新AI上下文
        self.context_manager.update_context(symbol, record)
        
        # 记录关键事件
        for event in key_events:
            self.context_manager.add_key_event(symbol, event)
        
        logger.info(f"完成 {symbol} 分析")
        return record
    
    def _run_strategy_analysis(self, prices: List[StockPrice]) -> Optional[Dict[str, Any]]:
        """执行策略分析"""
        if len(prices) < 20:
            return None
        
        try:
            price_data = [{
                'open': p.open,
                'high': p.high,
                'low': p.low,
                'close': p.close,
                'volume': p.volume
            } for p in prices]
            
            return self.strategy_analyzer.analyze(price_data)
        except Exception as e:
            logger.error(f"策略分析失败: {e}")
            return None
    
    def _analyze_analyst_ratings(self, ratings: List[Dict]) -> Dict[str, Any]:
        """分析机构评级"""
        if not ratings:
            return {"summary": "暂无机构评级"}
        
        buy_count = 0
        hold_count = 0
        sell_count = 0
        
        for r in ratings[:30]:
            rating = r.get('rating', '').lower()
            if 'buy' in rating or 'outperform' in rating or 'overweight' in rating:
                buy_count += 1
            elif 'sell' in rating or 'underperform' in rating or 'underweight' in rating:
                sell_count += 1
            else:
                hold_count += 1
        
        total = buy_count + hold_count + sell_count
        if total == 0:
            return {"summary": "暂无机构评级"}
        
        buy_pct = buy_count / total * 100
        sell_pct = sell_count / total * 100
        
        if buy_pct >= 60:
            consensus = "偏多"
        elif sell_pct >= 40:
            consensus = "偏空"
        else:
            consensus = "中性"
        
        return {
            "total": total,
            "buy": buy_count,
            "hold": hold_count,
            "sell": sell_count,
            "consensus": consensus,
            "buy_pct": buy_pct,
            "summary": f"机构评级: {buy_count}买入/{hold_count}持有/{sell_count}卖出 (共识:{consensus})"
        }
    
    def _analyze_earnings(self, earnings: List[Dict]) -> Dict[str, Any]:
        """分析财报数据"""
        if not earnings:
            return {"summary": "暂无财报数据"}
        
        beat_count = 0
        miss_count = 0
        recent_surprises = []
        
        for e in earnings[:8]:
            surprise = e.get('surprise', 0)
            if surprise and surprise > 0:
                beat_count += 1
            elif surprise and surprise < 0:
                miss_count += 1
            
            if e.get('surprisePercent'):
                recent_surprises.append(e.get('surprisePercent', 0))
        
        avg_surprise = sum(recent_surprises) / len(recent_surprises) if recent_surprises else 0
        
        if beat_count >= miss_count * 2:
            trend = "持续超预期"
        elif miss_count >= beat_count * 2:
            trend = "持续不及预期"
        else:
            trend = "符合预期"
        
        return {
            "total": len(earnings),
            "beat": beat_count,
            "miss": miss_count,
            "avg_surprise": avg_surprise,
            "trend": trend,
            "summary": f"财报: {beat_count}次超预期/{miss_count}次不及预期 (趋势:{trend})"
        }
    
    def _calculate_technical_indicators(
        self, 
        prices: List[StockPrice]
    ) -> Optional[TechnicalIndicators]:
        """计算技术指标"""
        if len(prices) < 20:
            return None
        
        close_prices = [p.close for p in prices]
        high_prices = [p.high for p in prices]
        low_prices = [p.low for p in prices]
        
        try:
            # 计算简单移动平均线
            sma_5 = sum(close_prices[-5:]) / 5 if len(close_prices) >= 5 else None
            sma_10 = sum(close_prices[-10:]) / 10 if len(close_prices) >= 10 else None
            sma_20 = sum(close_prices[-20:]) / 20
            sma_50 = sum(close_prices[-50:]) / 50 if len(close_prices) >= 50 else None
            sma_200 = sum(close_prices[-200:]) / 200 if len(close_prices) >= 200 else None
            
            # 计算RSI
            rsi = self._calculate_rsi(close_prices, 14)
            
            # 计算MACD
            macd, signal, hist = self._calculate_macd(close_prices)
            
            # 计算布林带
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            
            # 计算ATR
            atr = self._calculate_atr(high_prices, low_prices, close_prices)
            
            return TechnicalIndicators(
                sma_5=round(sma_5, 2) if sma_5 else None,
                sma_10=round(sma_10, 2) if sma_10 else None,
                sma_20=round(sma_20, 2),
                sma_50=round(sma_50, 2) if sma_50 else None,
                sma_200=round(sma_200, 2) if sma_200 else None,
                rsi_14=round(rsi, 2) if rsi else None,
                macd=round(macd, 4) if macd else None,
                macd_signal=round(signal, 4) if signal else None,
                macd_hist=round(hist, 4) if hist else None,
                bollinger_upper=round(bb_upper, 2) if bb_upper else None,
                bollinger_middle=round(bb_middle, 2) if bb_middle else None,
                bollinger_lower=round(bb_lower, 2) if bb_lower else None,
                atr=round(atr, 2) if atr else None
            )
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return None
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """计算RSI"""
        if len(prices) < period + 1:
            return None
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(
        self, 
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """计算MACD"""
        if len(prices) < slow + signal:
            return None, None, None
        
        # 计算EMA
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_values = [sum(data[:period]) / period]
            for price in data[period:]:
                ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
            return ema_values[-1]
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # 计算MACD历史用于信号线
        macd_history = []
        for i in range(slow, len(prices) + 1):
            ef = ema(prices[:i], fast)
            es = ema(prices[:i], slow)
            macd_history.append(ef - es)
        
        signal_line = ema(macd_history, signal) if len(macd_history) >= signal else macd_line
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(
        self, 
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple:
        """计算布林带"""
        if len(prices) < period:
            return None, None, None
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / period
        
        variance = sum((p - middle) ** 2 for p in recent_prices) / period
        std = variance ** 0.5
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    def _calculate_atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> Optional[float]:
        """计算ATR"""
        if len(highs) < period + 1:
            return None
        
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        return sum(tr_list[-period:]) / period
    
    def _extract_key_events(
        self,
        news_list: List[StockNews],
        earnings: List[Dict],
        ratings: List[Dict]
    ) -> List[str]:
        """提取关键事件"""
        events = []
        
        # 从新闻中提取高影响事件
        important_keywords = ["收购", "合并", "拆分", "退市", "诉讼", "召回", 
                            "破产", "裁员", "上调", "下调", "merger", "acquisition",
                            "lawsuit", "recall", "layoff", "upgrade", "downgrade"]
        
        for news in news_list[:20]:
            title_lower = news.title.lower()
            for kw in important_keywords:
                if kw in title_lower:
                    events.append(f"[新闻] {news.title[:50]}")
                    break
        
        # 财报事件
        for e in earnings[:2]:
            if e.get('surprisePercent'):
                surprise = e.get('surprisePercent', 0)
                if abs(surprise) > 10:
                    events.append(f"[财报] {e.get('date')}: EPS惊喜 {surprise:+.1f}%")
        
        # 评级变动
        for r in ratings[:5]:
            action = r.get('action', '').lower()
            if 'upgrade' in action or 'downgrade' in action:
                events.append(f"[评级] {r.get('firm')}: {r.get('rating')} ({action})")
        
        return events[:10]
    
    def _generate_ai_summary(
        self,
        symbol: str,
        price: MarketOverview,
        technical: Optional[TechnicalIndicators],
        sentiment: Dict,
        strategy: Optional[Dict],
        ratings: Dict,
        earnings: Dict,
        key_events: List[str]
    ) -> str:
        """生成AI分析摘要"""
        lines = [
            f"【{symbol} 综合分析报告】",
            f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "=" * 50,
            "【价格信息】",
            f"当前价格: ${price.current_price:.2f}",
            f"涨跌幅: {price.change:+.2f} ({price.change_percent:+.2f}%)",
            f"成交量: {price.volume:,}",
            "",
        ]
        
        # 策略分析结果
        if strategy and "error" not in strategy:
            lines.extend([
                "=" * 50,
                "【量化策略分析】",
                f"综合信号: {strategy.get('signal', 'N/A')}",
                f"置信度: {strategy.get('confidence', 0):.1f}%",
                f"综合评分: {strategy.get('score', 0):+.2f}",
                "",
                "趋势分析:",
            ])
            
            trend = strategy.get('trend', {})
            for ind in trend.get('indicators', []):
                lines.append(f"  • {ind['name']}: {ind['desc']}")
            
            lines.extend(["", "动量分析:"])
            momentum = strategy.get('momentum', {})
            for ind in momentum.get('indicators', []):
                lines.append(f"  • {ind['name']}: {ind['desc']}")
            
            lines.extend(["", "波动率分析:"])
            volatility = strategy.get('volatility', {})
            for ind in volatility.get('indicators', []):
                lines.append(f"  • {ind['name']}: {ind['desc']}")
            lines.append("")
        
        # 技术指标
        if technical:
            lines.extend([
                "=" * 50,
                "【技术指标】",
                f"RSI(14): {technical.rsi_14 or 'N/A'}",
                f"MACD: {technical.macd or 'N/A'}",
                f"布林带: {technical.bollinger_lower:.2f} - {technical.bollinger_middle:.2f} - {technical.bollinger_upper:.2f}" if technical.bollinger_middle else "",
                f"ATR(14): {technical.atr or 'N/A'}",
                "",
            ])
        
        # 市场情绪
        sentiment_label = {"positive": "偏多", "negative": "偏空", "neutral": "中性"}
        lines.extend([
            "=" * 50,
            "【市场情绪】",
            f"舆情分析: {sentiment_label.get(sentiment.get('overall', 'neutral'), '中性')}",
            f"情绪得分: {sentiment.get('score', 0):+.2f}",
            f"新闻数量: {sentiment.get('total', 0)}条",
            "",
        ])
        
        # 机构评级
        lines.extend([
            "=" * 50,
            "【机构评级】",
            ratings.get('summary', '暂无数据'),
            "",
        ])
        
        # 财报分析
        lines.extend([
            "=" * 50,
            "【财报分析】",
            earnings.get('summary', '暂无数据'),
            "",
        ])
        
        # 关键事件
        if key_events:
            lines.extend([
                "=" * 50,
                "【关键事件】",
            ])
            for event in key_events[:5]:
                lines.append(f"  • {event}")
            lines.append("")
        
        lines.extend([
            "=" * 50,
            "【分析说明】",
            "本分析基于公开数据和技术指标生成，仅供参考，不构成投资建议。",
        ])
        
        return "\n".join(lines)
    
    def _save_analysis_record(self, record: AnalysisRecord) -> None:
        """保存分析记录到数据库"""
        try:
            db_record = AnalysisRecordDB(
                symbol=record.symbol,
                timestamp=record.timestamp,
                current_price=record.current_price,
                price_change=record.price_change,
                price_change_percent=record.price_change_percent,
                volume=record.volume,
                technical_data=record.technical.model_dump_json() if record.technical else None,
                sentiment_score=record.sentiment_score,
                sentiment_label=record.sentiment_label,
                news_count=record.news_count,
                ai_summary=record.ai_summary
            )
            self.db.add(db_record)
            self.db.commit()
            logger.info(f"分析记录已保存: {record.symbol}")
        except Exception as e:
            logger.error(f"保存分析记录失败: {e}")
            self.db.rollback()
