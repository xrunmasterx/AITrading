"""
Streamlit 用户界面
简洁明了的股票分析可视化界面 - v2.0 实时监控版
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import time
from typing import Optional, List, Dict, Any

# 页面配置
st.set_page_config(
    page_title="AITrading - 量化分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session_state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'active_symbol' not in st.session_state:
    st.session_state.active_symbol = None
if 'monitor_active' not in st.session_state:
    st.session_state.monitor_active = False
if 'last_price_update' not in st.session_state:
    st.session_state.last_price_update = None
if 'realtime_prices' not in st.session_state:
    st.session_state.realtime_prices = []

# 自定义CSS样式
st.markdown("""
<style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 卡片样式 */
    .data-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border-left: 4px solid #00D4AA;
    }
    
    .news-item {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 3px solid #00D4AA;
    }
    
    .news-item.negative { border-left-color: #FF6B6B; }
    .news-item.neutral { border-left-color: #888; }
    
    .earning-card {
        background: #16213e;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .rating-bullish { color: #00D4AA; font-weight: bold; }
    .rating-bearish { color: #FF6B6B; font-weight: bold; }
    
    /* 标签样式 */
    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 5px;
    }
    .tag-earnings { background: #4CAF50; color: white; }
    .tag-filing { background: #2196F3; color: white; }
    .tag-rating { background: #FF9800; color: white; }
    .tag-insider { background: #9C27B0; color: white; }
</style>
""", unsafe_allow_html=True)


# ==================== 数据获取函数 ====================

def run_async(coro):
    """运行异步函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def fetch_stock_data(symbol: str, period: str = "3mo"):
    """获取股票数据（不缓存，确保周期变化生效）"""
    from app.services.data_fetcher import DataFetcher
    
    async def _fetch():
        fetcher = DataFetcher()
        return await fetcher.fetch_all_data(symbol, period=period)
    
    return run_async(_fetch())


def analyze_stock(symbol: str):
    """执行股票分析"""
    from app.database.db import get_db_manager
    from app.services.analyzer import StockAnalyzer
    
    async def _analyze():
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            analyzer = StockAnalyzer(session)
            return await analyzer.analyze(symbol)
    
    return run_async(_analyze())


def get_ai_context(symbol: str) -> str:
    """获取AI上下文"""
    from app.database.db import get_db_manager
    from app.utils.ai_context import AIContextManager
    
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        context_manager = AIContextManager(session)
        return context_manager.get_prompt_context(symbol)


# ==================== v2.0 自选股和监控函数 ====================

def get_watchlist() -> List[Dict]:
    """获取自选股列表"""
    from app.database.db import get_db_manager
    from app.services.watchlist import WatchlistService
    
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        service = WatchlistService(session)
        return service.get_watchlist_summary()


def add_to_watchlist(symbol: str, name: str = "") -> bool:
    """添加股票到自选股"""
    from app.database.db import get_db_manager
    from app.services.watchlist import WatchlistService
    
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        service = WatchlistService(session)
        result = service.add_stock(symbol, name)
        return result is not None


def remove_from_watchlist(symbol: str) -> bool:
    """从自选股删除"""
    from app.database.db import get_db_manager
    from app.services.watchlist import WatchlistService
    
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        service = WatchlistService(session)
        return service.remove_stock(symbol)


def set_active_stock(symbol: str):
    """设置激活的股票"""
    from app.database.db import get_db_manager
    from app.services.watchlist import WatchlistService
    
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        service = WatchlistService(session)
        service.set_active(symbol)


def fetch_realtime_price(symbol: str) -> Optional[Dict]:
    """获取实时价格快照"""
    from app.services.data_fetcher import DataFetcher
    
    async def _fetch():
        fetcher = DataFetcher()
        return await fetcher.fetch_realtime_snapshot(symbol)
    
    return run_async(_fetch())


def get_alert_config(symbol: str) -> Dict:
    """获取预警配置"""
    from app.database.db import get_db_manager
    from app.services.alert_checker import AlertChecker
    
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        checker = AlertChecker(session)
        return checker.get_alert_status(symbol)


def save_alert_config(symbol: str, upper: Optional[float], lower: Optional[float], email: str):
    """保存预警配置"""
    from app.database.db import get_db_manager
    from app.services.alert_checker import AlertChecker
    
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        checker = AlertChecker(session)
        checker.set_alert_config(symbol, upper, lower, email)


def check_price_alert(symbol: str, price: float):
    """检查价格预警"""
    from app.database.db import get_db_manager
    from app.services.alert_checker import AlertChecker
    
    async def _check():
        db_manager = get_db_manager()
        with db_manager.get_session() as session:
            checker = AlertChecker(session)
            return await checker.check_and_notify(symbol, price)
    
    return run_async(_check())


# ==================== 图表函数 ====================

def create_candlestick_chart(prices: List, symbol: str, period: str):
    """创建K线图"""
    if not prices:
        return None
    
    df = pd.DataFrame([{
        'date': p.timestamp,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume
    } for p in prices])
    
    # 创建子图：K线 + 成交量
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#00D4AA',
            decreasing_line_color='#FF6B6B'
        ),
        row=1, col=1
    )
    
    # 添加均线
    if len(df) >= 5:
        df['MA5'] = df['close'].rolling(window=5).mean()
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MA5'], name='MA5',
                      line=dict(color='#FF6B6B', width=1)),
            row=1, col=1
        )
    
    if len(df) >= 10:
        df['MA10'] = df['close'].rolling(window=10).mean()
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MA10'], name='MA10',
                      line=dict(color='#FFD700', width=1)),
            row=1, col=1
        )
    
    if len(df) >= 20:
        df['MA20'] = df['close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MA20'], name='MA20',
                      line=dict(color='#00A8CC', width=1)),
            row=1, col=1
        )
    
    if len(df) >= 60:
        df['MA60'] = df['close'].rolling(window=60).mean()
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['MA60'], name='MA60',
                      line=dict(color='#9C27B0', width=1)),
            row=1, col=1
        )
    
    # 成交量
    colors = ['#00D4AA' if row['close'] >= row['open'] else '#FF6B6B' 
              for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(x=df['date'], y=df['volume'], name='成交量',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # 布局设置
    period_text = {"1mo": "1个月", "3mo": "3个月", "6mo": "6个月", "1y": "1年", "2y": "2年"}.get(period, period)
    fig.update_layout(
        title=f'{symbol} K线图 ({period_text})',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_rating_chart(ratings: List[Dict]) -> Optional[go.Figure]:
    """创建分析师评级图表"""
    if not ratings:
        return None
    
    # 统计评级
    rating_counts = {"buy": 0, "hold": 0, "sell": 0, "strongBuy": 0, "strongSell": 0}
    for r in ratings[:20]:
        rating = r.get('rating', '').lower()
        if 'strong' in rating and 'buy' in rating:
            rating_counts['strongBuy'] += 1
        elif 'buy' in rating or 'outperform' in rating:
            rating_counts['buy'] += 1
        elif 'sell' in rating or 'underperform' in rating:
            rating_counts['sell'] += 1
        elif 'strong' in rating and 'sell' in rating:
            rating_counts['strongSell'] += 1
        else:
            rating_counts['hold'] += 1
    
    labels = ['强烈买入', '买入', '持有', '卖出', '强烈卖出']
    values = [rating_counts['strongBuy'], rating_counts['buy'], rating_counts['hold'], 
              rating_counts['sell'], rating_counts['strongSell']]
    colors = ['#00D4AA', '#4CAF50', '#FFD700', '#FF9800', '#FF6B6B']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.5, marker_colors=colors,
        textinfo='label+value'
    )])
    
    fig.update_layout(
        title="分析师评级分布",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig


# ==================== 主界面 ====================

def main():
    """主函数 - v2.0"""
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## 📈 AITrading v2.0")
        st.markdown("*实时监控 | 自动预警*")
        st.markdown("---")
        
        # ========== 自选股管理 ==========
        st.markdown("### 📋 自选股管理")
        
        # 添加股票
        col_add1, col_add2 = st.columns([3, 1])
        with col_add1:
            new_symbol = st.text_input(
                "添加股票",
                placeholder="如 AAPL, BABA",
                label_visibility="collapsed"
            )
        with col_add2:
            if st.button("➕", use_container_width=True, help="添加到自选"):
                if new_symbol:
                    if add_to_watchlist(new_symbol):
                        st.success(f"已添加 {new_symbol.upper()}")
                        st.rerun()
        
        # 自选股列表
        watchlist = get_watchlist()
        if watchlist:
            st.markdown("**自选股列表:**")
            for stock in watchlist:
                col_s, col_x = st.columns([4, 1])
                with col_s:
                    btn_label = f"{'★ ' if stock['is_active'] else ''}{stock['symbol']}"
                    if stock['name']:
                        btn_label += f" ({stock['name'][:6]})"
                    if st.button(btn_label, key=f"stock_{stock['symbol']}", use_container_width=True):
                        st.session_state.active_symbol = stock['symbol']
                        set_active_stock(stock['symbol'])
                        st.rerun()
                with col_x:
                    if st.button("×", key=f"del_{stock['symbol']}", help="删除"):
                        remove_from_watchlist(stock['symbol'])
                        if st.session_state.active_symbol == stock['symbol']:
                            st.session_state.active_symbol = None
                        st.rerun()
        else:
            st.info("暂无自选股，请添加")
        
        st.markdown("---")
        
        # ========== 实时监控控制 ==========
        st.markdown("### ⚡ 实时监控")
        
        from app.config import settings
        
        # 刷新间隔设置
        interval = st.slider(
            "刷新间隔(秒)",
            min_value=30, max_value=300, value=settings.realtime_interval,
            step=10, help="建议60秒以避免API限速"
        )
        
        # 监控开关
        if st.session_state.active_symbol:
            if st.session_state.monitor_active:
                if st.button("⏹️ 停止监控", use_container_width=True, type="secondary"):
                    st.session_state.monitor_active = False
                    st.rerun()
                st.success(f"🟢 正在监控: {st.session_state.active_symbol}")
            else:
                if st.button("▶️ 启动监控", use_container_width=True, type="primary"):
                    st.session_state.monitor_active = True
                    st.session_state.realtime_prices = []
                    st.rerun()
        else:
            st.warning("请先选择股票")
        
        st.markdown("---")
        
        # ========== 价格预警设置 ==========
        st.markdown("### 🔔 价格预警")
        
        if st.session_state.active_symbol:
            alert_config = get_alert_config(st.session_state.active_symbol)
            
            alert_upper = st.number_input(
                "上限价格",
                value=float(alert_config.get('upper_limit') or 0),
                min_value=0.0, step=0.1,
                help="价格达到或超过此值时通知"
            )
            
            alert_lower = st.number_input(
                "下限价格",
                value=float(alert_config.get('lower_limit') or 0),
                min_value=0.0, step=0.1,
                help="价格低于或等于此值时通知"
            )
            
            alert_email = st.text_input(
                "接收邮箱",
                value=alert_config.get('email') or settings.email_default_recipient,
                placeholder="your@email.com"
            )
            
            if st.button("💾 保存预警设置", use_container_width=True):
                save_alert_config(
                    st.session_state.active_symbol,
                    alert_upper if alert_upper > 0 else None,
                    alert_lower if alert_lower > 0 else None,
                    alert_email
                )
                st.success("预警设置已保存")
            
            if alert_config.get('configured'):
                status = "✅ 已配置" if alert_config.get('enabled') else "⏸️ 已暂停"
                st.info(status)
        else:
            st.info("选择股票后设置预警")
        
        st.markdown("---")
        
        # ========== 系统状态 ==========
        st.markdown("### ⚙️ 系统状态")
        st.success("✅ 数据库已连接")
        st.caption("📡 yfinance + Finnhub")
        st.caption("⚠️ 数据可能延迟15分钟")
        
        # ========== 传统模式（兼容）==========
        with st.expander("📊 传统分析模式"):
            symbol = st.text_input(
                "股票代码",
                value=st.session_state.active_symbol or "",
                key="manual_symbol"
            )
            
            period = st.selectbox(
                "时间周期",
                options=["1mo", "3mo", "6mo", "1y", "2y"],
                index=1,
                format_func=lambda x: {
                    "1mo": "1个月", "3mo": "3个月", 
                    "6mo": "6个月", "1y": "1年", "2y": "2年"
                }.get(x, x)
            )
            
            col1, col2 = st.columns(2)
            with col1:
                fetch_btn = st.button("🔄 获取", use_container_width=True)
            with col2:
                analyze_btn = st.button("🧠 分析", use_container_width=True)
    
    # 主内容区
    
    # 优先使用 active_symbol，否则使用手动输入的 symbol
    symbol = st.session_state.active_symbol or symbol
    
    if not symbol:
        show_welcome_page()
        return
    
    # 标准化股票代码
    from app.utils.helpers import parse_symbol
    std_symbol, market = parse_symbol(symbol)
    
    # ========== v2.0 实时状态栏 ==========
    if st.session_state.monitor_active and st.session_state.active_symbol == std_symbol:
        show_realtime_status_bar(std_symbol, interval)
    
    # 检测变化
    period_changed = st.session_state.get('current_period') != period
    symbol_changed = st.session_state.get('current_symbol') != std_symbol
    
    # 获取数据
    if fetch_btn or 'stock_data' not in st.session_state or symbol_changed or period_changed:
        with st.spinner(f"正在获取 {std_symbol} 数据 (周期: {period})..."):
            try:
                data = fetch_stock_data(std_symbol, period=period)
                st.session_state['stock_data'] = data
                st.session_state['current_symbol'] = std_symbol
                st.session_state['current_period'] = period
                
                # 显示数据统计
                stats = []
                if data.get('price_history'):
                    stats.append(f"K线: {len(data['price_history'])}条")
                if data.get('news'):
                    stats.append(f"新闻: {len(data['news'])}条")
                if data.get('earnings'):
                    stats.append(f"财报: {len(data['earnings'])}条")
                if data.get('analyst_ratings'):
                    stats.append(f"评级: {len(data['analyst_ratings'])}条")
                
                if stats:
                    st.success(f"✅ 数据获取成功! {' | '.join(stats)}")
                else:
                    st.warning("⚠️ 数据获取受限，请检查网络或稍后重试")
            except Exception as e:
                st.error(f"❌ 获取数据失败: {e}")
                return
    
    data = st.session_state.get('stock_data', {})
    if not data:
        st.warning("暂无数据，请点击获取数据按钮")
        return
    
    # 执行分析
    if analyze_btn:
        with st.spinner("正在分析..."):
            try:
                result = analyze_stock(std_symbol)
                if result:
                    st.session_state['analysis_result'] = result
                    st.success("✅ 分析完成!")
            except Exception as e:
                st.error(f"❌ 分析失败: {e}")
    
    # ========== 显示数据 ==========
    show_stock_header(data, std_symbol, market)
    show_price_metrics(data)
    
    st.markdown("---")
    
    # 使用标签页组织内容
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 K线图表", "📰 新闻资讯", "📈 财报数据", "🎯 分析师评级", "🧠 AI分析"
    ])
    
    with tab1:
        show_chart_tab(data, std_symbol, period)
    
    with tab2:
        show_news_tab(data)
    
    with tab3:
        show_earnings_tab(data)
    
    with tab4:
        show_ratings_tab(data)
    
    with tab5:
        show_analysis_tab(data, std_symbol)


def show_realtime_status_bar(symbol: str, interval: int):
    """显示实时监控状态栏"""
    # 获取实时价格
    price_data = fetch_realtime_price(symbol)
    
    if price_data:
        st.session_state.last_price_update = datetime.now()
        st.session_state.realtime_prices.append(price_data)
        
        # 限制存储的价格点数量
        if len(st.session_state.realtime_prices) > 100:
            st.session_state.realtime_prices = st.session_state.realtime_prices[-100:]
        
        # 检查预警
        check_price_alert(symbol, price_data['price'])
        
        # 状态栏
        price = price_data['price']
        change = price_data['change']
        change_pct = price_data['change_percent']
        
        color = "#00D4AA" if change >= 0 else "#FF6B6B"
        arrow = "▲" if change >= 0 else "▼"
        
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
                    border-radius: 10px; padding: 15px; margin-bottom: 20px;
                    display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 20px;">
                <div>
                    <span style="font-size: 1.5rem; font-weight: bold;">{symbol}</span>
                </div>
                <div>
                    <span style="font-size: 2rem; font-weight: bold; color: {color};">
                        ${price:.2f}
                    </span>
                </div>
                <div>
                    <span style="font-size: 1.2rem; color: {color};">
                        {arrow} {change:+.2f} ({change_pct:+.2f}%)
                    </span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="color: #00D4AA;">🟢 监控中</div>
                <div style="color: #888;">
                    ⏱️ {interval}秒后刷新
                </div>
                <div style="color: #888; font-size: 0.8rem;">
                    更新: {price_data['timestamp'].strftime('%H:%M:%S')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 自动刷新（使用 Streamlit 的 rerun 机制）
        time.sleep(interval)
        st.rerun()
    else:
        st.warning("⚠️ 无法获取实时数据，请检查网络")


def show_welcome_page():
    """显示欢迎页面 - v2.0"""
    st.markdown("""
    <div style="text-align: center; padding: 60px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 20px;">📈 AITrading v2.0</h1>
        <p style="font-size: 1.2rem; color: #888;">
            实时监控 | 自动预警 | 智能分析<br>
            在左侧添加自选股开始使用
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 快速开始")
    st.markdown("""
    1. **添加自选股** - 在左侧输入股票代码并点击 ➕
    2. **选择股票** - 点击列表中的股票名称
    3. **启动监控** - 点击「启动监控」按钮
    4. **设置预警** - 配置价格上下限和邮箱
    """)
    
    st.markdown("---")
    st.markdown("### 🔥 热门股票")
    cols = st.columns(4)
    examples = [("AAPL", "苹果"), ("BABA", "阿里巴巴"), ("NVDA", "英伟达"), ("TSLA", "特斯拉")]
    for col, (code, name) in zip(cols, examples):
        with col:
            if st.button(f"➕ {code}\n{name}", use_container_width=True):
                add_to_watchlist(code, name)
                st.session_state.active_symbol = code
                set_active_stock(code)
                st.rerun()


def show_stock_header(data: Dict, symbol: str, market: str):
    """显示股票头部信息"""
    info = data.get('info')
    if info:
        st.markdown(f"## {symbol} - {info.name}")
        st.markdown(f"*{market} | {info.sector or '未知行业'} | {info.industry or ''}*")


def show_price_metrics(data: Dict):
    """显示价格指标"""
    current_price = data.get('current_price')
    if not current_price:
        return
    
    cols = st.columns(5)
    
    with cols[0]:
        delta = f"{current_price.change_percent:.2f}%" if hasattr(current_price, 'change_percent') else None
        st.metric("当前价格", f"${current_price.current_price:.2f}", delta)
    
    with cols[1]:
        st.metric("今日最高", f"${current_price.day_high:.2f}")
    
    with cols[2]:
        st.metric("今日最低", f"${current_price.day_low:.2f}")
    
    with cols[3]:
        vol = current_price.volume
        vol_str = f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol/1e3:.1f}K"
        st.metric("成交量", vol_str)
    
    with cols[4]:
        if hasattr(current_price, 'prev_close') and current_price.prev_close:
            st.metric("昨收", f"${current_price.prev_close:.2f}")


def show_chart_tab(data: Dict, symbol: str, period: str):
    """K线图表标签页"""
    prices = data.get('price_history', [])
    
    if prices:
        st.markdown(f"**数据范围**: {prices[0].timestamp.strftime('%Y-%m-%d')} 至 {prices[-1].timestamp.strftime('%Y-%m-%d')} | **共 {len(prices)} 条数据**")
        
        fig = create_candlestick_chart(prices, symbol, period)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # 显示基本面信息
        info = data.get('info')
        if info:
            st.markdown("### 📋 基本面信息")
            info_cols = st.columns(4)
            with info_cols[0]:
                st.markdown(f"**市值:** {info.format_market_cap()}")
            with info_cols[1]:
                st.markdown(f"**PE:** {info.pe_ratio or 'N/A'}")
            with info_cols[2]:
                st.markdown(f"**PB:** {info.pb_ratio or 'N/A'}")
            with info_cols[3]:
                div_yield = f"{info.dividend_yield*100:.2f}%" if info.dividend_yield else "N/A"
                st.markdown(f"**股息率:** {div_yield}")
    else:
        st.info("暂无K线数据")


def show_news_tab(data: Dict):
    """新闻资讯标签页"""
    news_list = data.get('news', [])
    filings = data.get('sec_filings', [])
    insider = data.get('insider_transactions', [])
    
    # 统计
    st.markdown(f"### 📰 资讯汇总 (共 {len(news_list) + len(filings) + len(insider)} 条)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 新闻动态")
        if news_list:
            from app.services.sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            news_list = analyzer.analyze_news_list(news_list)
            sentiment_stats = analyzer.get_overall_sentiment(news_list)
            
            # 情绪统计
            sentiment_text = {"positive": "🟢 偏多", "negative": "🔴 偏空", "neutral": "⚪ 中性"}
            st.markdown(f"**整体情绪:** {sentiment_text.get(sentiment_stats.get('overall', 'neutral'))} | "
                       f"**得分:** {sentiment_stats.get('score', 0):+.2f} | "
                       f"**总数:** {len(news_list)}条")
            
            st.markdown("---")
            
            for news in news_list[:30]:
                sentiment_class = news.sentiment or "neutral"
                sentiment_icon = {"positive": "📈", "negative": "📉", "neutral": "➖"}.get(sentiment_class, "➖")
                
                st.markdown(f"""
                <div class="news-item {sentiment_class}">
                    <div style="font-size: 0.75rem; color: #888;">
                        {sentiment_icon} {news.source} | {news.published_at.strftime('%Y-%m-%d %H:%M')}
                    </div>
                    <div style="margin-top: 4px;">
                        <a href="{news.url}" target="_blank" style="color: #E8E8E8; text-decoration: none;">
                            {news.title}
                        </a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂无新闻数据")
    
    with col2:
        # SEC文件
        st.markdown("#### 📄 SEC文件")
        if filings:
            for f in filings[:10]:
                form_type = f.get('form', 'N/A')
                filed_date = f.get('filedDate', '')[:10]
                st.markdown(f"- **{form_type}** - {filed_date}")
        else:
            st.info("暂无SEC文件")
        
        # 内部交易
        st.markdown("#### 👤 内部人交易")
        if insider:
            for t in insider[:10]:
                name = t.get('name', 'Unknown')[:15]
                action = "买入" if t.get('transactionType') == 'buy' else "卖出"
                shares = t.get('shares', 0)
                st.markdown(f"- **{name}** {action} {shares:,}股")
        else:
            st.info("暂无内部人交易")


def show_earnings_tab(data: Dict):
    """财报数据标签页 - 增强版"""
    earnings = data.get('earnings', [])
    financials = data.get('financials', {})
    
    # 子标签页
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "📊 EPS数据", "📑 资产负债表", "💰 现金流", "🏛️ 机构持有"
    ])
    
    with sub_tab1:
        st.markdown(f"### 财报EPS (共 {len(earnings)} 条)")
        if earnings:
            df_data = []
            for e in earnings[:12]:
                df_data.append({
                    '日期': e.get('date', 'N/A'),
                    '实际EPS': e.get('reported_eps') or e.get('eps_actual', 'N/A'),
                    '预期EPS': e.get('eps_estimate', 'N/A'),
                    '惊喜%': f"{e.get('surprise', 0):.1f}%" if e.get('surprise') else 'N/A'
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无EPS数据")
    
    with sub_tab2:
        balance_sheet = financials.get('balance_sheet', [])
        st.markdown(f"### 资产负债表 (共 {len(balance_sheet)} 条)")
        if balance_sheet:
            for bs in balance_sheet:
                with st.expander(f"📅 {bs.get('date', 'N/A')}", expanded=len(balance_sheet) <= 2):
                    col1, col2 = st.columns(2)
                    with col1:
                        assets = bs.get('total_assets')
                        st.metric("总资产", f"${assets/1e9:.2f}B" if assets else "N/A")
                        cash = bs.get('cash')
                        st.metric("现金", f"${cash/1e9:.2f}B" if cash else "N/A")
                    with col2:
                        liab = bs.get('total_liabilities')
                        st.metric("总负债", f"${liab/1e9:.2f}B" if liab else "N/A")
                        equity = bs.get('total_equity')
                        st.metric("股东权益", f"${equity/1e9:.2f}B" if equity else "N/A")
        else:
            st.info("暂无资产负债表数据")
    
    with sub_tab3:
        cashflow = financials.get('cashflow', [])
        income = financials.get('income_stmt', [])
        st.markdown(f"### 现金流 & 收入 (共 {len(cashflow) + len(income)} 条)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**现金流**")
            if cashflow:
                for cf in cashflow[:3]:
                    st.markdown(f"**{cf.get('date', 'N/A')}**")
                    op = cf.get('operating_cashflow')
                    fcf = cf.get('free_cashflow')
                    st.write(f"- 经营现金流: ${op/1e9:.2f}B" if op else "- 经营现金流: N/A")
                    st.write(f"- 自由现金流: ${fcf/1e9:.2f}B" if fcf else "- 自由现金流: N/A")
            else:
                st.info("暂无现金流数据")
        
        with col2:
            st.markdown("**收入报表**")
            if income:
                for inc in income[:3]:
                    st.markdown(f"**{inc.get('date', 'N/A')}**")
                    rev = inc.get('total_revenue')
                    net = inc.get('net_income')
                    st.write(f"- 总收入: ${rev/1e9:.2f}B" if rev else "- 总收入: N/A")
                    st.write(f"- 净利润: ${net/1e9:.2f}B" if net else "- 净利润: N/A")
            else:
                st.info("暂无收入数据")
    
    with sub_tab4:
        holders = financials.get('institutional_holders', [])
        analysis = financials.get('analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏛️ 机构持有者 (Top {min(len(holders), 10)})")
            if holders:
                for h in holders[:10]:
                    holder = h.get('holder', 'Unknown')[:25]
                    pct = h.get('percent_out', 0)
                    shares = h.get('shares', 0)
                    st.markdown(f"- **{holder}**: {pct:.2f}% ({shares/1e6:.2f}M股)")
            else:
                st.info("暂无机构持有者数据")
        
        with col2:
            st.markdown("### 📊 分析师目标价")
            targets = analysis.get('price_targets', {})
            if targets:
                st.metric("目标价均值", f"${targets.get('mean', 0):.2f}")
                st.write(f"- 最低: ${targets.get('low', 0):.2f}")
                st.write(f"- 最高: ${targets.get('high', 0):.2f}")
                st.write(f"- 中位数: ${targets.get('median', 0):.2f}")
            else:
                st.info("暂无目标价数据")
            
            st.markdown("### 📈 增长预估")
            growth = analysis.get('growth_estimates', {})
            if growth:
                st.write(f"- 本季度: {growth.get('current_qtr', 'N/A')}")
                st.write(f"- 下季度: {growth.get('next_qtr', 'N/A')}")
                st.write(f"- 本年度: {growth.get('current_year', 'N/A')}")
                st.write(f"- 下年度: {growth.get('next_year', 'N/A')}")
            else:
                st.info("暂无增长预估")


def show_ratings_tab(data: Dict):
    """分析师评级标签页"""
    ratings = data.get('analyst_ratings', [])
    
    st.markdown(f"### 🎯 分析师评级 (共 {len(ratings)} 条)")
    
    if ratings:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 评级分布图
            fig = create_rating_chart(ratings)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 评级列表
            st.markdown("#### 最新评级")
            for r in ratings[:15]:
                date = r.get('date', '')[:10]
                firm = r.get('firm', 'Unknown')[:20]
                rating = r.get('rating', 'N/A')
                action = r.get('action', '')
                
                # 评级颜色
                rating_class = "rating-bullish" if 'buy' in rating.lower() else "rating-bearish" if 'sell' in rating.lower() else ""
                
                st.markdown(f"""
                <div class="earning-card">
                    <div style="font-size: 0.75rem; color: #888;">{date} | {firm}</div>
                    <div class="{rating_class}">{rating}</div>
                    <div style="font-size: 0.8rem; color: #aaa;">{action}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("暂无分析师评级数据")


def show_analysis_tab(data: Dict, symbol: str):
    """AI分析标签页"""
    
    # 快速策略分析（不需要点击按钮）
    st.markdown("### 📊 量化策略分析")
    
    prices = data.get('price_history', [])
    if len(prices) >= 20:
        from app.services.strategy import ComprehensiveAnalyzer
        
        price_data = [{
            'open': p.open, 'high': p.high, 'low': p.low,
            'close': p.close, 'volume': p.volume
        } for p in prices]
        
        analyzer = ComprehensiveAnalyzer()
        strategy_result = analyzer.analyze(price_data)
        
        if "error" not in strategy_result:
            # 信号展示
            signal = strategy_result.get('signal', 'N/A')
            confidence = strategy_result.get('confidence', 0)
            score = strategy_result.get('score', 0)
            
            signal_colors = {
                "强烈买入": "#00D4AA", "买入": "#4CAF50",
                "持有": "#FFD700", "卖出": "#FF9800", "强烈卖出": "#FF6B6B"
            }
            signal_color = signal_colors.get(signal, "#888")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                        border-radius: 10px; padding: 20px; margin-bottom: 20px;
                        border-left: 5px solid {signal_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.9rem; color: #888;">综合信号</div>
                        <div style="font-size: 2rem; color: {signal_color}; font-weight: bold;">{signal}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.9rem; color: #888;">置信度</div>
                        <div style="font-size: 1.5rem; color: #E8E8E8;">{confidence:.1f}%</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.9rem; color: #888;">综合评分</div>
                        <div style="font-size: 1.5rem; color: {'#00D4AA' if score > 0 else '#FF6B6B' if score < 0 else '#FFD700'};">{score:+.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 三个策略详情
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend = strategy_result.get('trend', {})
                st.markdown(f"**📈 趋势分析** - {trend.get('signal', 'N/A')}")
                for ind in trend.get('indicators', []):
                    st.markdown(f"• {ind['name']}: {ind['desc']}")
            
            with col2:
                momentum = strategy_result.get('momentum', {})
                st.markdown(f"**⚡ 动量分析** - {momentum.get('signal', 'N/A')}")
                for ind in momentum.get('indicators', []):
                    st.markdown(f"• {ind['name']}: {ind['desc']}")
            
            with col3:
                volatility = strategy_result.get('volatility', {})
                st.markdown(f"**📉 波动率分析** - {volatility.get('signal', 'N/A')}")
                for ind in volatility.get('indicators', []):
                    st.markdown(f"• {ind['name']}: {ind['desc']}")
        else:
            st.warning(strategy_result.get('error', '策略分析失败'))
    else:
        st.info(f"数据不足（{len(prices)}条），至少需要20条K线数据进行策略分析")
    
    st.markdown("---")
    
    # 完整分析结果
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧠 完整分析报告")
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            st.text_area("分析报告", result.ai_summary, height=400, label_visibility="collapsed")
        else:
            st.info("点击侧边栏「分析」按钮执行完整分析并保存记录")
    
    with col2:
        st.markdown("### 📝 历史上下文")
        try:
            context = get_ai_context(symbol)
            st.text_area("历史分析记录", context, height=400, label_visibility="collapsed")
        except Exception:
            st.info("暂无历史分析记录")


if __name__ == "__main__":
    main()
