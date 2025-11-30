"""
舆情分析模块
分析新闻和社交媒体的情绪倾向
"""

import re
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from loguru import logger

from app.models.stock import StockNews


class SentimentAnalyzer:
    """舆情分析器"""
    
    def __init__(self):
        # 简单的情绪词典（英文）
        self.positive_words = {
            'surge', 'jump', 'gain', 'rise', 'up', 'growth', 'profit', 'beat',
            'exceed', 'strong', 'bullish', 'rally', 'boom', 'soar', 'breakthrough',
            'upgrade', 'outperform', 'buy', 'positive', 'optimistic', 'record',
            'high', 'success', 'win', 'best', 'improve', 'increase', 'expand'
        }
        
        self.negative_words = {
            'drop', 'fall', 'decline', 'down', 'loss', 'miss', 'weak', 'bearish',
            'crash', 'plunge', 'slump', 'tumble', 'downgrade', 'underperform',
            'sell', 'negative', 'pessimistic', 'low', 'fail', 'worst', 'cut',
            'decrease', 'shrink', 'warning', 'risk', 'concern', 'fear', 'trouble'
        }
        
        # 中文情绪词典
        self.positive_words_cn = {
            '上涨', '大涨', '暴涨', '涨停', '突破', '新高', '利好', '增长',
            '盈利', '超预期', '看好', '买入', '推荐', '强势', '反弹', '回升',
            '创新', '领先', '优秀', '成功'
        }
        
        self.negative_words_cn = {
            '下跌', '大跌', '暴跌', '跌停', '破位', '新低', '利空', '下降',
            '亏损', '不及预期', '看空', '卖出', '减持', '弱势', '回调', '下滑',
            '风险', '警告', '担忧', '失败'
        }
    
    def analyze_text(self, text: str) -> Tuple[str, float]:
        """
        分析文本情绪
        
        Args:
            text: 要分析的文本
            
        Returns:
            (sentiment, score): 情绪类型和得分
            - sentiment: positive/negative/neutral
            - score: -1.0 到 1.0
        """
        if not text:
            return "neutral", 0.0
        
        text_lower = text.lower()
        
        # 计算英文情绪得分
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # 计算中文情绪得分
        pos_count += sum(1 for word in self.positive_words_cn if word in text)
        neg_count += sum(1 for word in self.negative_words_cn if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return "neutral", 0.0
        
        # 计算得分 (-1 到 1)
        score = (pos_count - neg_count) / total
        
        # 确定情绪类型
        if score > 0.2:
            sentiment = "positive"
        elif score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return sentiment, round(score, 3)
    
    def analyze_news(self, news: StockNews) -> StockNews:
        """
        分析单条新闻的情绪
        
        Args:
            news: 新闻对象
            
        Returns:
            带有情绪分析结果的新闻对象
        """
        # 合并标题和摘要进行分析
        text = f"{news.title} {news.summary}"
        sentiment, score = self.analyze_text(text)
        
        news.sentiment = sentiment
        news.sentiment_score = score
        
        return news
    
    def analyze_news_list(self, news_list: List[StockNews]) -> List[StockNews]:
        """
        批量分析新闻情绪
        
        Args:
            news_list: 新闻列表
            
        Returns:
            带有情绪分析结果的新闻列表
        """
        analyzed = []
        for news in news_list:
            analyzed.append(self.analyze_news(news))
        
        logger.info(f"完成 {len(analyzed)} 条新闻情绪分析")
        return analyzed
    
    def get_overall_sentiment(self, news_list: List[StockNews]) -> Dict:
        """
        获取新闻的整体情绪概览
        
        Args:
            news_list: 新闻列表
            
        Returns:
            情绪统计信息
        """
        if not news_list:
            return {
                "overall": "neutral",
                "score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total": 0
            }
        
        pos_count = 0
        neg_count = 0
        neutral_count = 0
        total_score = 0.0
        
        for news in news_list:
            if news.sentiment == "positive":
                pos_count += 1
            elif news.sentiment == "negative":
                neg_count += 1
            else:
                neutral_count += 1
            
            if news.sentiment_score is not None:
                total_score += news.sentiment_score
        
        total = len(news_list)
        avg_score = total_score / total if total > 0 else 0.0
        
        # 确定整体情绪
        if avg_score > 0.1:
            overall = "positive"
        elif avg_score < -0.1:
            overall = "negative"
        else:
            overall = "neutral"
        
        return {
            "overall": overall,
            "score": round(avg_score, 3),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neutral_count,
            "total": total,
            "positive_ratio": round(pos_count / total * 100, 1) if total > 0 else 0
        }
    
    def generate_sentiment_summary(self, news_list: List[StockNews]) -> str:
        """
        生成舆情摘要文本
        
        Args:
            news_list: 新闻列表
            
        Returns:
            舆情摘要字符串
        """
        stats = self.get_overall_sentiment(news_list)
        
        if stats['total'] == 0:
            return "暂无新闻数据，无法生成舆情分析。"
        
        sentiment_text = {
            "positive": "偏多",
            "negative": "偏空", 
            "neutral": "中性"
        }
        
        positive_ratio = stats.get('positive_ratio', 0)
        
        summary = (
            f"舆情概览: {sentiment_text.get(stats['overall'], '中性')} "
            f"(得分: {stats['score']:+.2f})\n"
            f"新闻统计: 共{stats['total']}条 - "
            f"看多{stats['positive_count']}条({positive_ratio}%) / "
            f"看空{stats['negative_count']}条 / "
            f"中性{stats['neutral_count']}条"
        )
        
        # 添加最近的关键新闻
        key_news = self._extract_key_news(news_list)
        if key_news:
            summary += "\n\n关键新闻:"
            for news in key_news[:3]:
                sentiment_icon = {
                    "positive": "📈",
                    "negative": "📉",
                    "neutral": "➖"
                }.get(news.sentiment, "➖")
                summary += f"\n{sentiment_icon} {news.title[:60]}..."
        
        return summary
    
    def _extract_key_news(self, news_list: List[StockNews]) -> List[StockNews]:
        """
        提取关键新闻（情绪倾向明显的）
        
        Args:
            news_list: 新闻列表
            
        Returns:
            关键新闻列表
        """
        # 按情绪得分绝对值排序
        sorted_news = sorted(
            news_list,
            key=lambda x: abs(x.sentiment_score or 0),
            reverse=True
        )
        
        return sorted_news[:5]
    
    def detect_key_events(self, news_list: List[StockNews]) -> List[str]:
        """
        检测新闻中的关键事件
        
        Args:
            news_list: 新闻列表
            
        Returns:
            关键事件列表
        """
        key_events = []
        
        # 关键词模式
        patterns = {
            "财报": r"(earnings|财报|季报|年报|业绩)",
            "收购": r"(acquire|acquisition|merger|收购|并购|合并)",
            "分拆": r"(split|spinoff|分拆|拆分)",
            "派息": r"(dividend|派息|分红|股息)",
            "评级": r"(upgrade|downgrade|rating|评级|上调|下调)",
            "裁员": r"(layoff|cut jobs|裁员|裁减)",
            "新产品": r"(launch|release|新产品|发布|上市)",
            "监管": r"(regulation|SEC|证监会|监管|调查)"
        }
        
        for news in news_list:
            text = f"{news.title} {news.summary}".lower()
            
            for event_type, pattern in patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    event = f"[{news.published_at.strftime('%m-%d')}] {event_type}: {news.title[:40]}"
                    if event not in key_events:
                        key_events.append(event)
        
        return key_events[:10]  # 最多返回10个关键事件


# 创建全局实例
def create_sentiment_analyzer() -> SentimentAnalyzer:
    """创建舆情分析器实例"""
    return SentimentAnalyzer()

