"""
量化分析策略模块
包含多种技术分析策略和综合评分系统
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from loguru import logger


class Signal(Enum):
    """交易信号"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


@dataclass
class IndicatorResult:
    """指标计算结果"""
    name: str
    value: float
    signal: Signal
    description: str
    weight: float = 1.0


@dataclass
class StrategyResult:
    """策略分析结果"""
    strategy_name: str
    signal: Signal
    confidence: float  # 0-100
    indicators: List[IndicatorResult]
    description: str
    

class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> Optional[float]:
        """简单移动平均线"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    @staticmethod
    def ema(prices: List[float], period: int) -> Optional[float]:
        """指数移动平均线"""
        if len(prices) < period:
            return None
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """相对强弱指数"""
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
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """MACD指标"""
        if len(prices) < slow + signal:
            return None, None, None
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None, None, None
        
        macd_line = ema_fast - ema_slow
        
        # 计算MACD历史用于信号线
        macd_history = []
        for i in range(slow, len(prices) + 1):
            ef = TechnicalIndicators.ema(prices[:i], fast)
            es = TechnicalIndicators.ema(prices[:i], slow)
            if ef and es:
                macd_history.append(ef - es)
        
        signal_line = TechnicalIndicators.ema(macd_history, signal) if len(macd_history) >= signal else macd_line
        histogram = macd_line - signal_line if signal_line else 0
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """布林带"""
        if len(prices) < period:
            return None, None, None
        
        recent = prices[-period:]
        middle = sum(recent) / period
        variance = sum((p - middle) ** 2 for p in recent) / period
        std = variance ** 0.5
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """平均真实范围"""
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
    
    @staticmethod
    def stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
        """随机指标 KDJ"""
        if len(closes) < k_period:
            return None, None
        
        highest = max(highs[-k_period:])
        lowest = min(lows[-k_period:])
        
        if highest == lowest:
            k = 50
        else:
            k = ((closes[-1] - lowest) / (highest - lowest)) * 100
        
        # 简化D值计算
        d = k  # 实际应该是K的移动平均
        
        return k, d
    
    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """平均趋向指数"""
        if len(highs) < period + 1:
            return None
        
        # 简化计算
        tr_sum = 0
        dm_plus_sum = 0
        dm_minus_sum = 0
        
        for i in range(1, min(period + 1, len(highs))):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_sum += tr
            
            dm_plus = highs[i] - highs[i-1] if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
            dm_minus = lows[i-1] - lows[i] if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
            
            dm_plus_sum += max(0, dm_plus)
            dm_minus_sum += max(0, dm_minus)
        
        if tr_sum == 0:
            return 0
        
        di_plus = (dm_plus_sum / tr_sum) * 100
        di_minus = (dm_minus_sum / tr_sum) * 100
        
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
        
        return dx
    
    @staticmethod
    def obv(closes: List[float], volumes: List[int]) -> Optional[float]:
        """能量潮指标"""
        if len(closes) < 2:
            return None
        
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return obv
    
    @staticmethod
    def williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """威廉指标"""
        if len(closes) < period:
            return None
        
        highest = max(highs[-period:])
        lowest = min(lows[-period:])
        
        if highest == lowest:
            return -50
        
        return ((highest - closes[-1]) / (highest - lowest)) * -100
    
    @staticmethod
    def cci(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Optional[float]:
        """顺势指标"""
        if len(closes) < period:
            return None
        
        tp_list = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(-period, 0)]
        tp = tp_list[-1]
        tp_sma = sum(tp_list) / period
        
        mean_dev = sum(abs(t - tp_sma) for t in tp_list) / period
        
        if mean_dev == 0:
            return 0
        
        return (tp - tp_sma) / (0.015 * mean_dev)


class TrendStrategy:
    """趋势跟踪策略"""
    
    def analyze(self, closes: List[float], highs: List[float] = None, lows: List[float] = None) -> StrategyResult:
        """执行趋势分析"""
        indicators = []
        
        # 1. 均线系统
        sma5 = TechnicalIndicators.sma(closes, 5)
        sma10 = TechnicalIndicators.sma(closes, 10)
        sma20 = TechnicalIndicators.sma(closes, 20)
        sma60 = TechnicalIndicators.sma(closes, 60)
        
        current_price = closes[-1]
        
        # 均线多头/空头排列
        ma_score = 0
        if sma5 and sma10 and sma20:
            if sma5 > sma10 > sma20:
                ma_score = 2  # 多头排列
                ma_signal = Signal.BUY
                ma_desc = "均线多头排列，短期趋势向上"
            elif sma5 < sma10 < sma20:
                ma_score = -2  # 空头排列
                ma_signal = Signal.SELL
                ma_desc = "均线空头排列，短期趋势向下"
            else:
                ma_score = 0
                ma_signal = Signal.HOLD
                ma_desc = "均线交织，趋势不明"
            
            indicators.append(IndicatorResult(
                name="均线系统",
                value=ma_score,
                signal=ma_signal,
                description=ma_desc,
                weight=1.5
            ))
        
        # 价格与均线位置
        if sma20:
            price_vs_ma = (current_price - sma20) / sma20 * 100
            if price_vs_ma > 5:
                pos_signal = Signal.BUY
                pos_desc = f"价格高于MA20 {price_vs_ma:.1f}%，强势"
            elif price_vs_ma < -5:
                pos_signal = Signal.SELL
                pos_desc = f"价格低于MA20 {price_vs_ma:.1f}%，弱势"
            else:
                pos_signal = Signal.HOLD
                pos_desc = f"价格接近MA20，观望"
            
            indicators.append(IndicatorResult(
                name="价格位置",
                value=price_vs_ma,
                signal=pos_signal,
                description=pos_desc,
                weight=1.0
            ))
        
        # 2. ADX趋势强度
        if highs and lows:
            adx = TechnicalIndicators.adx(highs, lows, closes)
            if adx:
                if adx > 25:
                    adx_signal = Signal.BUY if ma_score > 0 else Signal.SELL
                    adx_desc = f"ADX={adx:.1f}，趋势强劲"
                else:
                    adx_signal = Signal.HOLD
                    adx_desc = f"ADX={adx:.1f}，趋势较弱"
                
                indicators.append(IndicatorResult(
                    name="ADX趋势强度",
                    value=adx,
                    signal=adx_signal,
                    description=adx_desc,
                    weight=1.0
                ))
        
        # 计算综合信号
        signal, confidence = self._calculate_signal(indicators)
        
        return StrategyResult(
            strategy_name="趋势跟踪策略",
            signal=signal,
            confidence=confidence,
            indicators=indicators,
            description=f"基于均线系统和趋势强度的分析"
        )
    
    def _calculate_signal(self, indicators: List[IndicatorResult]) -> Tuple[Signal, float]:
        """计算综合信号"""
        if not indicators:
            return Signal.HOLD, 50.0
        
        score = 0
        total_weight = 0
        
        signal_scores = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.HOLD: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2
        }
        
        for ind in indicators:
            score += signal_scores[ind.signal] * ind.weight
            total_weight += ind.weight
        
        avg_score = score / total_weight if total_weight > 0 else 0
        
        if avg_score >= 1.5:
            signal = Signal.STRONG_BUY
        elif avg_score >= 0.5:
            signal = Signal.BUY
        elif avg_score <= -1.5:
            signal = Signal.STRONG_SELL
        elif avg_score <= -0.5:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        confidence = min(100, max(0, 50 + avg_score * 25))
        
        return signal, confidence


class MomentumStrategy:
    """动量策略"""
    
    def analyze(self, closes: List[float], highs: List[float] = None, lows: List[float] = None, volumes: List[int] = None) -> StrategyResult:
        """执行动量分析"""
        indicators = []
        
        # 1. RSI
        rsi = TechnicalIndicators.rsi(closes)
        if rsi:
            if rsi > 70:
                rsi_signal = Signal.SELL
                rsi_desc = f"RSI={rsi:.1f}，超买区域"
            elif rsi < 30:
                rsi_signal = Signal.BUY
                rsi_desc = f"RSI={rsi:.1f}，超卖区域"
            elif rsi > 50:
                rsi_signal = Signal.BUY
                rsi_desc = f"RSI={rsi:.1f}，偏强势"
            else:
                rsi_signal = Signal.SELL
                rsi_desc = f"RSI={rsi:.1f}，偏弱势"
            
            indicators.append(IndicatorResult(
                name="RSI",
                value=rsi,
                signal=rsi_signal,
                description=rsi_desc,
                weight=1.5
            ))
        
        # 2. MACD
        macd, signal_line, histogram = TechnicalIndicators.macd(closes)
        if macd is not None and signal_line is not None:
            if macd > signal_line and histogram > 0:
                macd_signal = Signal.BUY
                macd_desc = f"MACD金叉，柱状图为正"
            elif macd < signal_line and histogram < 0:
                macd_signal = Signal.SELL
                macd_desc = f"MACD死叉，柱状图为负"
            else:
                macd_signal = Signal.HOLD
                macd_desc = f"MACD信号不明确"
            
            indicators.append(IndicatorResult(
                name="MACD",
                value=histogram or 0,
                signal=macd_signal,
                description=macd_desc,
                weight=1.5
            ))
        
        # 3. 随机指标
        if highs and lows:
            k, d = TechnicalIndicators.stochastic(highs, lows, closes)
            if k is not None:
                if k > 80:
                    kd_signal = Signal.SELL
                    kd_desc = f"K={k:.1f}，超买"
                elif k < 20:
                    kd_signal = Signal.BUY
                    kd_desc = f"K={k:.1f}，超卖"
                else:
                    kd_signal = Signal.HOLD
                    kd_desc = f"K={k:.1f}，中性"
                
                indicators.append(IndicatorResult(
                    name="KDJ",
                    value=k,
                    signal=kd_signal,
                    description=kd_desc,
                    weight=1.0
                ))
        
        # 4. 威廉指标
        if highs and lows:
            wr = TechnicalIndicators.williams_r(highs, lows, closes)
            if wr is not None:
                if wr > -20:
                    wr_signal = Signal.SELL
                    wr_desc = f"W%R={wr:.1f}，超买"
                elif wr < -80:
                    wr_signal = Signal.BUY
                    wr_desc = f"W%R={wr:.1f}，超卖"
                else:
                    wr_signal = Signal.HOLD
                    wr_desc = f"W%R={wr:.1f}，中性"
                
                indicators.append(IndicatorResult(
                    name="威廉%R",
                    value=wr,
                    signal=wr_signal,
                    description=wr_desc,
                    weight=0.8
                ))
        
        # 5. CCI
        if highs and lows:
            cci = TechnicalIndicators.cci(highs, lows, closes)
            if cci is not None:
                if cci > 100:
                    cci_signal = Signal.BUY
                    cci_desc = f"CCI={cci:.1f}，强势"
                elif cci < -100:
                    cci_signal = Signal.SELL
                    cci_desc = f"CCI={cci:.1f}，弱势"
                else:
                    cci_signal = Signal.HOLD
                    cci_desc = f"CCI={cci:.1f}，中性"
                
                indicators.append(IndicatorResult(
                    name="CCI",
                    value=cci,
                    signal=cci_signal,
                    description=cci_desc,
                    weight=0.8
                ))
        
        # 计算综合信号
        signal, confidence = self._calculate_signal(indicators)
        
        return StrategyResult(
            strategy_name="动量策略",
            signal=signal,
            confidence=confidence,
            indicators=indicators,
            description="基于RSI、MACD、KDJ等动量指标的分析"
        )
    
    def _calculate_signal(self, indicators: List[IndicatorResult]) -> Tuple[Signal, float]:
        """计算综合信号"""
        if not indicators:
            return Signal.HOLD, 50.0
        
        score = 0
        total_weight = 0
        
        signal_scores = {
            Signal.STRONG_BUY: 2, Signal.BUY: 1, Signal.HOLD: 0,
            Signal.SELL: -1, Signal.STRONG_SELL: -2
        }
        
        for ind in indicators:
            score += signal_scores[ind.signal] * ind.weight
            total_weight += ind.weight
        
        avg_score = score / total_weight if total_weight > 0 else 0
        
        if avg_score >= 1.5:
            signal = Signal.STRONG_BUY
        elif avg_score >= 0.5:
            signal = Signal.BUY
        elif avg_score <= -1.5:
            signal = Signal.STRONG_SELL
        elif avg_score <= -0.5:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        confidence = min(100, max(0, 50 + avg_score * 25))
        return signal, confidence


class VolatilityStrategy:
    """波动率策略"""
    
    def analyze(self, closes: List[float], highs: List[float] = None, lows: List[float] = None) -> StrategyResult:
        """执行波动率分析"""
        indicators = []
        current_price = closes[-1]
        
        # 1. 布林带
        upper, middle, lower = TechnicalIndicators.bollinger_bands(closes)
        if upper and middle and lower:
            bb_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
            
            if current_price > upper:
                bb_signal = Signal.SELL
                bb_desc = f"价格突破布林带上轨，可能超买"
            elif current_price < lower:
                bb_signal = Signal.BUY
                bb_desc = f"价格跌破布林带下轨，可能超卖"
            elif bb_position > 0.8:
                bb_signal = Signal.SELL
                bb_desc = f"价格接近布林带上轨，注意风险"
            elif bb_position < 0.2:
                bb_signal = Signal.BUY
                bb_desc = f"价格接近布林带下轨，关注机会"
            else:
                bb_signal = Signal.HOLD
                bb_desc = f"价格在布林带中轨附近"
            
            indicators.append(IndicatorResult(
                name="布林带",
                value=bb_position * 100,
                signal=bb_signal,
                description=bb_desc,
                weight=1.5
            ))
            
            # 布林带宽度
            bb_width = (upper - lower) / middle * 100 if middle else 0
            if bb_width < 5:
                width_desc = f"布林带收窄({bb_width:.1f}%)，可能即将突破"
            elif bb_width > 20:
                width_desc = f"布林带扩张({bb_width:.1f}%)，波动加大"
            else:
                width_desc = f"布林带宽度正常({bb_width:.1f}%)"
            
            indicators.append(IndicatorResult(
                name="布林带宽度",
                value=bb_width,
                signal=Signal.HOLD,
                description=width_desc,
                weight=0.5
            ))
        
        # 2. ATR
        if highs and lows:
            atr = TechnicalIndicators.atr(highs, lows, closes)
            if atr:
                atr_percent = atr / current_price * 100
                
                if atr_percent > 5:
                    atr_desc = f"ATR={atr:.2f}({atr_percent:.1f}%)，高波动"
                elif atr_percent < 2:
                    atr_desc = f"ATR={atr:.2f}({atr_percent:.1f}%)，低波动"
                else:
                    atr_desc = f"ATR={atr:.2f}({atr_percent:.1f}%)，正常波动"
                
                indicators.append(IndicatorResult(
                    name="ATR",
                    value=atr_percent,
                    signal=Signal.HOLD,
                    description=atr_desc,
                    weight=1.0
                ))
        
        # 计算综合信号
        signal, confidence = self._calculate_signal(indicators)
        
        return StrategyResult(
            strategy_name="波动率策略",
            signal=signal,
            confidence=confidence,
            indicators=indicators,
            description="基于布林带和ATR的波动率分析"
        )
    
    def _calculate_signal(self, indicators: List[IndicatorResult]) -> Tuple[Signal, float]:
        """计算综合信号"""
        if not indicators:
            return Signal.HOLD, 50.0
        
        score = 0
        total_weight = 0
        
        signal_scores = {
            Signal.STRONG_BUY: 2, Signal.BUY: 1, Signal.HOLD: 0,
            Signal.SELL: -1, Signal.STRONG_SELL: -2
        }
        
        for ind in indicators:
            score += signal_scores[ind.signal] * ind.weight
            total_weight += ind.weight
        
        avg_score = score / total_weight if total_weight > 0 else 0
        
        if avg_score >= 1:
            signal = Signal.BUY
        elif avg_score <= -1:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        confidence = min(100, max(0, 50 + avg_score * 20))
        return signal, confidence


class ComprehensiveAnalyzer:
    """综合分析器 - 整合多种策略"""
    
    def __init__(self):
        self.trend_strategy = TrendStrategy()
        self.momentum_strategy = MomentumStrategy()
        self.volatility_strategy = VolatilityStrategy()
    
    def analyze(self, prices: List[Dict]) -> Dict[str, Any]:
        """
        执行综合分析
        
        Args:
            prices: 价格数据列表，每个元素包含 open, high, low, close, volume
        
        Returns:
            综合分析结果
        """
        if not prices or len(prices) < 20:
            return {
                "error": "数据不足，至少需要20条K线数据",
                "signal": Signal.HOLD.value,
                "confidence": 0
            }
        
        # 提取数据
        closes = [p['close'] for p in prices]
        highs = [p['high'] for p in prices]
        lows = [p['low'] for p in prices]
        volumes = [p.get('volume', 0) for p in prices]
        
        # 执行各策略分析
        trend_result = self.trend_strategy.analyze(closes, highs, lows)
        momentum_result = self.momentum_strategy.analyze(closes, highs, lows, volumes)
        volatility_result = self.volatility_strategy.analyze(closes, highs, lows)
        
        # 综合评分
        strategy_weights = {
            "趋势": 0.4,
            "动量": 0.4,
            "波动率": 0.2
        }
        
        signal_scores = {
            Signal.STRONG_BUY: 2, Signal.BUY: 1, Signal.HOLD: 0,
            Signal.SELL: -1, Signal.STRONG_SELL: -2
        }
        
        total_score = (
            signal_scores[trend_result.signal] * strategy_weights["趋势"] +
            signal_scores[momentum_result.signal] * strategy_weights["动量"] +
            signal_scores[volatility_result.signal] * strategy_weights["波动率"]
        )
        
        # 确定最终信号
        if total_score >= 1.2:
            final_signal = Signal.STRONG_BUY
        elif total_score >= 0.4:
            final_signal = Signal.BUY
        elif total_score <= -1.2:
            final_signal = Signal.STRONG_SELL
        elif total_score <= -0.4:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD
        
        # 计算综合置信度
        avg_confidence = (
            trend_result.confidence * strategy_weights["趋势"] +
            momentum_result.confidence * strategy_weights["动量"] +
            volatility_result.confidence * strategy_weights["波动率"]
        )
        
        # 生成分析报告
        report = self._generate_report(
            closes[-1], trend_result, momentum_result, volatility_result,
            final_signal, avg_confidence, total_score
        )
        
        return {
            "signal": final_signal.value,
            "confidence": round(avg_confidence, 1),
            "score": round(total_score, 2),
            "trend": {
                "signal": trend_result.signal.value,
                "confidence": trend_result.confidence,
                "indicators": [{"name": i.name, "value": i.value, "signal": i.signal.value, "desc": i.description} 
                              for i in trend_result.indicators]
            },
            "momentum": {
                "signal": momentum_result.signal.value,
                "confidence": momentum_result.confidence,
                "indicators": [{"name": i.name, "value": i.value, "signal": i.signal.value, "desc": i.description}
                              for i in momentum_result.indicators]
            },
            "volatility": {
                "signal": volatility_result.signal.value,
                "confidence": volatility_result.confidence,
                "indicators": [{"name": i.name, "value": i.value, "signal": i.signal.value, "desc": i.description}
                              for i in volatility_result.indicators]
            },
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_report(self, current_price: float, trend: StrategyResult, 
                        momentum: StrategyResult, volatility: StrategyResult,
                        final_signal: Signal, confidence: float, score: float) -> str:
        """生成分析报告"""
        
        signal_emoji = {
            Signal.STRONG_BUY: "🟢🟢",
            Signal.BUY: "🟢",
            Signal.HOLD: "🟡",
            Signal.SELL: "🔴",
            Signal.STRONG_SELL: "🔴🔴"
        }
        
        lines = [
            "=" * 50,
            f"【综合分析报告】",
            f"当前价格: ${current_price:.2f}",
            f"综合信号: {signal_emoji[final_signal]} {final_signal.value}",
            f"置信度: {confidence:.1f}%",
            f"综合评分: {score:+.2f}",
            "",
            "【趋势分析】" + f" {trend.signal.value}",
        ]
        
        for ind in trend.indicators:
            lines.append(f"  • {ind.name}: {ind.description}")
        
        lines.extend([
            "",
            "【动量分析】" + f" {momentum.signal.value}",
        ])
        
        for ind in momentum.indicators:
            lines.append(f"  • {ind.name}: {ind.description}")
        
        lines.extend([
            "",
            "【波动率分析】" + f" {volatility.signal.value}",
        ])
        
        for ind in volatility.indicators:
            lines.append(f"  • {ind.name}: {ind.description}")
        
        lines.extend([
            "",
            "=" * 50,
            f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        
        return "\n".join(lines)


def create_comprehensive_analyzer() -> ComprehensiveAnalyzer:
    """创建综合分析器实例"""
    return ComprehensiveAnalyzer()


