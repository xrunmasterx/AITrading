"""
高级量化分析策略模块
包含多种技术分析、趋势分析和综合评分系统
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger

from app.models.stock import StockPrice


class TrendDirection(str, Enum):
    """趋势方向"""
    STRONG_UP = "strong_up"      # 强势上涨
    UP = "up"                     # 上涨
    SIDEWAYS = "sideways"         # 横盘
    DOWN = "down"                 # 下跌
    STRONG_DOWN = "strong_down"   # 强势下跌


class SignalStrength(str, Enum):
    """信号强度"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TechnicalSignal:
    """技术信号"""
    name: str
    value: float
    signal: SignalStrength
    description: str
    weight: float = 1.0


@dataclass
class AnalysisResult:
    """分析结果"""
    symbol: str
    timestamp: datetime
    
    # 综合评分 (-100 到 100)
    overall_score: float
    overall_signal: SignalStrength
    
    # 各维度分析
    technical_score: float
    trend_score: float
    momentum_score: float
    volume_score: float
    sentiment_score: float
    
    # 详细信号
    signals: List[TechnicalSignal]
    
    # 关键位置
    support_levels: List[float]
    resistance_levels: List[float]
    
    # 风险评估
    risk_level: str  # low/medium/high
    volatility: float
    
    # 建议
    recommendation: str
    key_points: List[str]


class AdvancedAnalyzer:
    """高级量化分析器"""
    
    def __init__(self):
        pass
    
    def analyze(
        self,
        prices: List[StockPrice],
        news_sentiment: Optional[float] = None,
        analyst_ratings: Optional[List[Dict]] = None,
        earnings: Optional[List[Dict]] = None
    ) -> Optional[AnalysisResult]:
        """
        执行全面分析
        
        Args:
            prices: 价格历史数据
            news_sentiment: 新闻舆情得分 (-1 到 1)
            analyst_ratings: 机构评级数据
            earnings: 财报数据
            
        Returns:
            分析结果
        """
        if not prices or len(prices) < 20:
            logger.warning("价格数据不足，无法进行完整分析")
            return None
        
        symbol = prices[0].symbol
        close_prices = [p.close for p in prices]
        high_prices = [p.high for p in prices]
        low_prices = [p.low for p in prices]
        volumes = [p.volume for p in prices]
        
        # 1. 技术指标分析
        tech_signals, tech_score = self._analyze_technical(close_prices, high_prices, low_prices, volumes)
        
        # 2. 趋势分析
        trend_score, trend_direction = self._analyze_trend(close_prices)
        
        # 3. 动量分析
        momentum_score = self._analyze_momentum(close_prices)
        
        # 4. 量价分析
        volume_score = self._analyze_volume(close_prices, volumes)
        
        # 5. 舆情分析
        sentiment_score = self._process_sentiment(news_sentiment, analyst_ratings)
        
        # 6. 支撑/阻力位
        support, resistance = self._find_support_resistance(close_prices, high_prices, low_prices)
        
        # 7. 波动率和风险
        volatility = self._calculate_volatility(close_prices)
        risk_level = self._assess_risk(volatility, close_prices)
        
        # 8. 综合评分
        overall_score = self._calculate_overall_score(
            tech_score, trend_score, momentum_score, volume_score, sentiment_score
        )
        overall_signal = self._score_to_signal(overall_score)
        
        # 9. 生成建议
        recommendation, key_points = self._generate_recommendation(
            overall_score, tech_signals, trend_direction, 
            support, resistance, close_prices[-1]
        )
        
        return AnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            overall_signal=overall_signal,
            technical_score=tech_score,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            sentiment_score=sentiment_score,
            signals=tech_signals,
            support_levels=support,
            resistance_levels=resistance,
            risk_level=risk_level,
            volatility=volatility,
            recommendation=recommendation,
            key_points=key_points
        )
    
    def _analyze_technical(
        self, 
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float]
    ) -> Tuple[List[TechnicalSignal], float]:
        """技术指标分析"""
        signals = []
        
        # RSI分析
        rsi = self._calculate_rsi(closes, 14)
        if rsi:
            if rsi > 70:
                sig = TechnicalSignal("RSI", rsi, SignalStrength.SELL, f"RSI={rsi:.1f} 超买区域", 1.5)
            elif rsi > 60:
                sig = TechnicalSignal("RSI", rsi, SignalStrength.NEUTRAL, f"RSI={rsi:.1f} 偏强", 1.0)
            elif rsi < 30:
                sig = TechnicalSignal("RSI", rsi, SignalStrength.BUY, f"RSI={rsi:.1f} 超卖区域", 1.5)
            elif rsi < 40:
                sig = TechnicalSignal("RSI", rsi, SignalStrength.NEUTRAL, f"RSI={rsi:.1f} 偏弱", 1.0)
            else:
                sig = TechnicalSignal("RSI", rsi, SignalStrength.NEUTRAL, f"RSI={rsi:.1f} 中性", 0.5)
            signals.append(sig)
        
        # MACD分析
        macd, signal_line, histogram = self._calculate_macd(closes)
        if macd is not None:
            if histogram > 0 and macd > signal_line:
                sig = TechnicalSignal("MACD", histogram, SignalStrength.BUY, "MACD金叉，动能向上", 1.2)
            elif histogram < 0 and macd < signal_line:
                sig = TechnicalSignal("MACD", histogram, SignalStrength.SELL, "MACD死叉，动能向下", 1.2)
            else:
                sig = TechnicalSignal("MACD", histogram, SignalStrength.NEUTRAL, "MACD信号不明确", 0.5)
            signals.append(sig)
        
        # 均线分析
        ma_signal = self._analyze_moving_averages(closes)
        signals.append(ma_signal)
        
        # 布林带分析
        bb_signal = self._analyze_bollinger_bands(closes)
        if bb_signal:
            signals.append(bb_signal)
        
        # KDJ分析
        kdj_signal = self._analyze_kdj(closes, highs, lows)
        if kdj_signal:
            signals.append(kdj_signal)
        
        # 计算技术得分
        tech_score = self._signals_to_score(signals)
        
        return signals, tech_score
    
    def _analyze_trend(self, closes: List[float]) -> Tuple[float, TrendDirection]:
        """趋势分析"""
        if len(closes) < 50:
            return 0, TrendDirection.SIDEWAYS
        
        # 短期趋势 (5日)
        short_trend = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] else 0
        
        # 中期趋势 (20日)
        mid_trend = (closes[-1] - closes[-20]) / closes[-20] * 100 if closes[-20] else 0
        
        # 长期趋势 (50日)
        long_trend = (closes[-1] - closes[-50]) / closes[-50] * 100 if len(closes) >= 50 else mid_trend
        
        # 均线排列
        ma5 = sum(closes[-5:]) / 5
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20
        
        # 趋势得分
        trend_score = 0
        
        # 价格趋势贡献
        if short_trend > 3:
            trend_score += 20
        elif short_trend > 0:
            trend_score += 10
        elif short_trend < -3:
            trend_score -= 20
        elif short_trend < 0:
            trend_score -= 10
        
        if mid_trend > 5:
            trend_score += 25
        elif mid_trend > 0:
            trend_score += 15
        elif mid_trend < -5:
            trend_score -= 25
        elif mid_trend < 0:
            trend_score -= 15
        
        if long_trend > 10:
            trend_score += 30
        elif long_trend > 0:
            trend_score += 15
        elif long_trend < -10:
            trend_score -= 30
        elif long_trend < 0:
            trend_score -= 15
        
        # 均线排列贡献
        if ma5 > ma20 > ma50:
            trend_score += 25
            direction = TrendDirection.STRONG_UP if trend_score > 50 else TrendDirection.UP
        elif ma5 < ma20 < ma50:
            trend_score -= 25
            direction = TrendDirection.STRONG_DOWN if trend_score < -50 else TrendDirection.DOWN
        else:
            direction = TrendDirection.SIDEWAYS
        
        return max(-100, min(100, trend_score)), direction
    
    def _analyze_momentum(self, closes: List[float]) -> float:
        """动量分析"""
        if len(closes) < 14:
            return 0
        
        # ROC (Rate of Change)
        roc_5 = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] else 0
        roc_10 = (closes[-1] - closes[-10]) / closes[-10] * 100 if closes[-10] else 0
        
        # 动量加速度
        momentum_recent = closes[-1] - closes[-3]
        momentum_prev = closes[-3] - closes[-5]
        acceleration = momentum_recent - momentum_prev
        
        # 计算得分
        score = 0
        
        # ROC贡献
        if roc_5 > 5:
            score += 30
        elif roc_5 > 2:
            score += 15
        elif roc_5 < -5:
            score -= 30
        elif roc_5 < -2:
            score -= 15
        
        if roc_10 > 10:
            score += 25
        elif roc_10 > 5:
            score += 15
        elif roc_10 < -10:
            score -= 25
        elif roc_10 < -5:
            score -= 15
        
        # 加速度贡献
        if acceleration > 0:
            score += 20
        elif acceleration < 0:
            score -= 20
        
        return max(-100, min(100, score))
    
    def _analyze_volume(self, closes: List[float], volumes: List[float]) -> float:
        """量价分析"""
        if len(volumes) < 20:
            return 0
        
        # 平均成交量
        avg_vol = sum(volumes[-20:]) / 20
        recent_vol = sum(volumes[-5:]) / 5
        
        # 价格变化
        price_change = closes[-1] - closes[-5]
        
        score = 0
        
        # 量价配合
        vol_ratio = recent_vol / avg_vol if avg_vol else 1
        
        if price_change > 0:
            # 上涨
            if vol_ratio > 1.5:
                score += 40  # 放量上涨，强势
            elif vol_ratio > 1:
                score += 20  # 温和放量上涨
            elif vol_ratio < 0.7:
                score -= 10  # 缩量上涨，动能不足
        else:
            # 下跌
            if vol_ratio > 1.5:
                score -= 40  # 放量下跌，恐慌
            elif vol_ratio > 1:
                score -= 20  # 温和放量下跌
            elif vol_ratio < 0.7:
                score += 10  # 缩量下跌，抛压减轻
        
        # OBV趋势
        obv_trend = self._calculate_obv_trend(closes, volumes)
        score += obv_trend * 20
        
        return max(-100, min(100, score))
    
    def _process_sentiment(
        self, 
        news_sentiment: Optional[float],
        analyst_ratings: Optional[List[Dict]]
    ) -> float:
        """处理舆情和机构评级"""
        score = 0
        
        # 新闻舆情 (-1 到 1)
        if news_sentiment is not None:
            score += news_sentiment * 40  # 最多贡献40分
        
        # 机构评级
        if analyst_ratings:
            buy_count = 0
            sell_count = 0
            
            for rating in analyst_ratings:
                if rating.get('source') == 'finnhub_trend':
                    buy_count += rating.get('strong_buy', 0) + rating.get('buy', 0)
                    sell_count += rating.get('sell', 0) + rating.get('strong_sell', 0)
            
            total = buy_count + sell_count
            if total > 0:
                rating_score = (buy_count - sell_count) / total * 30
                score += rating_score
        
        return max(-100, min(100, score))
    
    def _find_support_resistance(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float]
    ) -> Tuple[List[float], List[float]]:
        """寻找支撑位和阻力位"""
        if len(closes) < 20:
            return [], []
        
        current_price = closes[-1]
        
        # 近期低点作为支撑
        recent_lows = sorted(lows[-60:] if len(lows) >= 60 else lows)[:5]
        supports = [l for l in recent_lows if l < current_price][:3]
        
        # 近期高点作为阻力
        recent_highs = sorted(highs[-60:] if len(highs) >= 60 else highs, reverse=True)[:5]
        resistances = [h for h in recent_highs if h > current_price][:3]
        
        # 添加均线作为动态支撑/阻力
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
        
        if ma20 < current_price and ma20 not in supports:
            supports.append(round(ma20, 2))
        elif ma20 > current_price and ma20 not in resistances:
            resistances.append(round(ma20, 2))
        
        if ma50:
            if ma50 < current_price and ma50 not in supports:
                supports.append(round(ma50, 2))
            elif ma50 > current_price and ma50 not in resistances:
                resistances.append(round(ma50, 2))
        
        return sorted(supports, reverse=True)[:3], sorted(resistances)[:3]
    
    def _calculate_volatility(self, closes: List[float]) -> float:
        """计算波动率"""
        if len(closes) < 20:
            return 0
        
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        
        # 标准差年化
        std = np.std(returns[-20:])
        annual_volatility = std * np.sqrt(252) * 100
        
        return round(annual_volatility, 2)
    
    def _assess_risk(self, volatility: float, closes: List[float]) -> str:
        """评估风险等级"""
        # 基于波动率
        if volatility > 50:
            return "high"
        elif volatility > 30:
            return "medium"
        else:
            return "low"
    
    def _calculate_overall_score(
        self,
        tech_score: float,
        trend_score: float,
        momentum_score: float,
        volume_score: float,
        sentiment_score: float
    ) -> float:
        """计算综合得分"""
        # 权重分配
        weights = {
            'technical': 0.25,
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'sentiment': 0.15
        }
        
        overall = (
            tech_score * weights['technical'] +
            trend_score * weights['trend'] +
            momentum_score * weights['momentum'] +
            volume_score * weights['volume'] +
            sentiment_score * weights['sentiment']
        )
        
        return round(overall, 1)
    
    def _score_to_signal(self, score: float) -> SignalStrength:
        """将分数转换为信号"""
        if score >= 50:
            return SignalStrength.STRONG_BUY
        elif score >= 20:
            return SignalStrength.BUY
        elif score <= -50:
            return SignalStrength.STRONG_SELL
        elif score <= -20:
            return SignalStrength.SELL
        else:
            return SignalStrength.NEUTRAL
    
    def _signals_to_score(self, signals: List[TechnicalSignal]) -> float:
        """将信号列表转换为得分"""
        if not signals:
            return 0
        
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return 0
        
        score = 0
        for s in signals:
            signal_score = {
                SignalStrength.STRONG_BUY: 100,
                SignalStrength.BUY: 50,
                SignalStrength.NEUTRAL: 0,
                SignalStrength.SELL: -50,
                SignalStrength.STRONG_SELL: -100
            }.get(s.signal, 0)
            
            score += signal_score * s.weight
        
        return score / total_weight
    
    def _generate_recommendation(
        self,
        overall_score: float,
        signals: List[TechnicalSignal],
        trend: TrendDirection,
        supports: List[float],
        resistances: List[float],
        current_price: float
    ) -> Tuple[str, List[str]]:
        """生成投资建议"""
        key_points = []
        
        # 主要建议
        if overall_score >= 50:
            recommendation = "强烈看多：多个技术指标发出买入信号，建议积极布局"
        elif overall_score >= 20:
            recommendation = "谨慎看多：整体偏多，但需关注回调风险，可逢低买入"
        elif overall_score <= -50:
            recommendation = "强烈看空：多个指标发出卖出信号，建议减仓或观望"
        elif overall_score <= -20:
            recommendation = "谨慎看空：整体偏空，注意止损，等待企稳信号"
        else:
            recommendation = "中性观望：信号不明确，建议等待方向明朗后再操作"
        
        # 关键点
        if trend == TrendDirection.STRONG_UP:
            key_points.append("📈 处于强势上涨趋势，均线多头排列")
        elif trend == TrendDirection.UP:
            key_points.append("📈 上涨趋势中，动能良好")
        elif trend == TrendDirection.STRONG_DOWN:
            key_points.append("📉 处于强势下跌趋势，均线空头排列")
        elif trend == TrendDirection.DOWN:
            key_points.append("📉 下跌趋势中，注意风险")
        else:
            key_points.append("➖ 横盘整理中，等待方向选择")
        
        # 支撑阻力
        if supports:
            key_points.append(f"📍 下方支撑位: {', '.join(f'${s:.2f}' for s in supports[:2])}")
        if resistances:
            key_points.append(f"📍 上方阻力位: {', '.join(f'${r:.2f}' for r in resistances[:2])}")
        
        # 技术信号摘要
        buy_signals = [s for s in signals if s.signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL]]
        
        if buy_signals:
            key_points.append(f"✅ 买入信号: {', '.join(s.name for s in buy_signals)}")
        if sell_signals:
            key_points.append(f"⚠️ 卖出信号: {', '.join(s.name for s in sell_signals)}")
        
        return recommendation, key_points
    
    # ==================== 技术指标计算 ====================
    
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
        
        return round(rsi, 1)
    
    def _calculate_macd(
        self, 
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """计算MACD"""
        if len(prices) < slow + signal:
            return None, None, None
        
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_val = sum(data[:period]) / period
            for price in data[period:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line * 0.85  # 简化
        histogram = macd_line - signal_line
        
        return round(macd_line, 4), round(signal_line, 4), round(histogram, 4)
    
    def _analyze_moving_averages(self, closes: List[float]) -> TechnicalSignal:
        """均线分析"""
        if len(closes) < 50:
            return TechnicalSignal("均线", 0, SignalStrength.NEUTRAL, "数据不足", 0.5)
        
        ma5 = sum(closes[-5:]) / 5
        ma10 = sum(closes[-10:]) / 10
        ma20 = sum(closes[-20:]) / 20
        ma50 = sum(closes[-50:]) / 50
        
        current = closes[-1]
        
        # 判断排列
        if current > ma5 > ma10 > ma20 > ma50:
            return TechnicalSignal("均线", current, SignalStrength.STRONG_BUY, "完美多头排列", 1.5)
        elif current > ma5 > ma20:
            return TechnicalSignal("均线", current, SignalStrength.BUY, "多头排列", 1.2)
        elif current < ma5 < ma10 < ma20 < ma50:
            return TechnicalSignal("均线", current, SignalStrength.STRONG_SELL, "完美空头排列", 1.5)
        elif current < ma5 < ma20:
            return TechnicalSignal("均线", current, SignalStrength.SELL, "空头排列", 1.2)
        else:
            return TechnicalSignal("均线", current, SignalStrength.NEUTRAL, "均线交织", 0.5)
    
    def _analyze_bollinger_bands(self, closes: List[float], period: int = 20) -> Optional[TechnicalSignal]:
        """布林带分析"""
        if len(closes) < period:
            return None
        
        recent = closes[-period:]
        middle = sum(recent) / period
        std = np.std(recent)
        
        upper = middle + 2 * std
        lower = middle - 2 * std
        
        current = closes[-1]
        
        # 判断位置
        position = (current - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        
        if current > upper:
            return TechnicalSignal("布林带", position, SignalStrength.SELL, "触及上轨，可能回调", 1.0)
        elif current < lower:
            return TechnicalSignal("布林带", position, SignalStrength.BUY, "触及下轨，可能反弹", 1.0)
        elif position > 0.8:
            return TechnicalSignal("布林带", position, SignalStrength.NEUTRAL, "接近上轨", 0.5)
        elif position < 0.2:
            return TechnicalSignal("布林带", position, SignalStrength.NEUTRAL, "接近下轨", 0.5)
        else:
            return TechnicalSignal("布林带", position, SignalStrength.NEUTRAL, "通道中部", 0.3)
    
    def _analyze_kdj(
        self, 
        closes: List[float], 
        highs: List[float], 
        lows: List[float],
        period: int = 9
    ) -> Optional[TechnicalSignal]:
        """KDJ分析"""
        if len(closes) < period:
            return None
        
        # 计算RSV
        highest = max(highs[-period:])
        lowest = min(lows[-period:])
        
        if highest == lowest:
            return None
        
        rsv = (closes[-1] - lowest) / (highest - lowest) * 100
        
        # 简化的K值
        k = rsv
        
        if k > 80:
            return TechnicalSignal("KDJ", k, SignalStrength.SELL, f"K={k:.0f} 超买区", 1.0)
        elif k < 20:
            return TechnicalSignal("KDJ", k, SignalStrength.BUY, f"K={k:.0f} 超卖区", 1.0)
        else:
            return TechnicalSignal("KDJ", k, SignalStrength.NEUTRAL, f"K={k:.0f} 中性", 0.5)
    
    def _calculate_obv_trend(self, closes: List[float], volumes: List[float]) -> float:
        """计算OBV趋势"""
        if len(closes) < 10:
            return 0
        
        obv = 0
        obv_history = []
        
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
            obv_history.append(obv)
        
        if len(obv_history) < 10:
            return 0
        
        # OBV趋势
        recent_obv = obv_history[-5:]
        prev_obv = obv_history[-10:-5]
        
        if sum(recent_obv) > sum(prev_obv):
            return 1  # 资金流入
        elif sum(recent_obv) < sum(prev_obv):
            return -1  # 资金流出
        else:
            return 0
    
    def generate_report(self, result: AnalysisResult) -> str:
        """生成分析报告文本"""
        lines = [
            f"{'='*50}",
            f"【{result.symbol} 量化分析报告】",
            f"分析时间: {result.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"{'='*50}",
            "",
            f"📊 综合评分: {result.overall_score:+.1f} ({result.overall_signal.value})",
            "",
            "【分项得分】",
            f"  技术面: {result.technical_score:+.1f}",
            f"  趋势: {result.trend_score:+.1f}",
            f"  动量: {result.momentum_score:+.1f}",
            f"  量价: {result.volume_score:+.1f}",
            f"  舆情: {result.sentiment_score:+.1f}",
            "",
            f"📈 波动率: {result.volatility:.1f}% (风险: {result.risk_level})",
            "",
            "【技术信号】"
        ]
        
        for sig in result.signals:
            icon = "✅" if sig.signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY] else (
                "⚠️" if sig.signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL] else "➖"
            )
            lines.append(f"  {icon} {sig.name}: {sig.description}")
        
        lines.extend([
            "",
            "【支撑/阻力】",
            f"  支撑位: {', '.join(f'${s:.2f}' for s in result.support_levels) or 'N/A'}",
            f"  阻力位: {', '.join(f'${r:.2f}' for r in result.resistance_levels) or 'N/A'}",
            "",
            "【投资建议】",
            f"  {result.recommendation}",
            "",
            "【关键要点】"
        ])
        
        for point in result.key_points:
            lines.append(f"  • {point}")
        
        lines.append(f"\n{'='*50}")
        
        return "\n".join(lines)


def create_advanced_analyzer() -> AdvancedAnalyzer:
    """创建高级分析器实例"""
    return AdvancedAnalyzer()

