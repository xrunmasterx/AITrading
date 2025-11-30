"""
FastAPI 主入口
提供RESTful API接口
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from loguru import logger
import sys

from app.config import settings
from app.database.db import get_db, get_db_manager

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    settings.logs_dir / "app_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG"
)

# 创建FastAPI应用
app = FastAPI(
    title="AITrading API",
    description="微型量化软件 - 美股/港股数据采集与分析",
    version="0.1.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("AITrading API 启动中...")
    # 初始化数据库
    db_manager = get_db_manager()
    logger.info("数据库初始化完成")


@app.get("/")
async def root():
    """根路由"""
    return {
        "name": "AITrading API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/stock/{symbol}")
async def get_stock_info(symbol: str, db: Session = Depends(get_db)):
    """
    获取股票基本信息
    
    - **symbol**: 股票代码，如 AAPL, 0700.HK
    """
    from app.services.data_fetcher import DataFetcher
    from app.utils.helpers import parse_symbol
    
    std_symbol, market = parse_symbol(symbol)
    fetcher = DataFetcher()
    
    try:
        info = await fetcher.get_stock_info(std_symbol)
        if info:
            return info.model_dump()
        raise HTTPException(status_code=404, detail=f"未找到股票: {symbol}")
    except Exception as e:
        logger.error(f"获取股票信息失败: {symbol}, 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/price")
async def get_stock_price(
    symbol: str,
    period: str = Query(default="1mo", description="时间周期: 1d, 5d, 1mo, 3mo, 6mo, 1y"),
    db: Session = Depends(get_db)
):
    """
    获取股票价格数据
    
    - **symbol**: 股票代码
    - **period**: 时间周期
    """
    from app.services.data_fetcher import DataFetcher
    from app.utils.helpers import parse_symbol
    
    std_symbol, market = parse_symbol(symbol)
    fetcher = DataFetcher()
    
    try:
        prices = await fetcher.get_price_history(std_symbol, period=period)
        return {
            "symbol": std_symbol,
            "market": market,
            "period": period,
            "count": len(prices),
            "data": [p.model_dump() for p in prices]
        }
    except Exception as e:
        logger.error(f"获取价格数据失败: {symbol}, 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/news")
async def get_stock_news(
    symbol: str,
    limit: int = Query(default=20, le=50),
    db: Session = Depends(get_db)
):
    """
    获取股票相关新闻
    
    - **symbol**: 股票代码
    - **limit**: 返回条数，最大50
    """
    from app.services.data_fetcher import DataFetcher
    from app.utils.helpers import parse_symbol
    
    std_symbol, market = parse_symbol(symbol)
    fetcher = DataFetcher()
    
    try:
        news = await fetcher.get_news(std_symbol, limit=limit)
        return {
            "symbol": std_symbol,
            "count": len(news),
            "news": [n.model_dump() for n in news]
        }
    except Exception as e:
        logger.error(f"获取新闻失败: {symbol}, 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/context")
async def get_ai_context(symbol: str, db: Session = Depends(get_db)):
    """
    获取股票的AI分析上下文
    
    - **symbol**: 股票代码
    """
    from app.utils.ai_context import AIContextManager
    from app.utils.helpers import parse_symbol
    
    std_symbol, _ = parse_symbol(symbol)
    context_manager = AIContextManager(db)
    
    context = context_manager.get_context(std_symbol)
    if context:
        return {
            "symbol": std_symbol,
            "context": context.to_prompt_context(),
            "analysis_count": context.analysis_count,
            "updated_at": context.updated_at.isoformat()
        }
    return {
        "symbol": std_symbol,
        "context": "暂无历史分析数据",
        "analysis_count": 0
    }


@app.post("/api/stock/{symbol}/note")
async def add_note(symbol: str, note: str, db: Session = Depends(get_db)):
    """
    添加用户备注
    
    - **symbol**: 股票代码
    - **note**: 备注内容
    """
    from app.utils.ai_context import AIContextManager
    from app.utils.helpers import parse_symbol
    
    std_symbol, _ = parse_symbol(symbol)
    context_manager = AIContextManager(db)
    
    # 确保上下文存在
    context_manager.get_or_create_context(std_symbol)
    context_manager.add_user_note(std_symbol, note)
    
    return {"status": "success", "message": f"备注已添加到 {std_symbol}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_debug
    )



