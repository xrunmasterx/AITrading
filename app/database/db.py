"""
数据库连接和管理
"""

from pathlib import Path
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from app.config import settings
from app.database.schemas import Base


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        
        # 确保数据目录存在
        if self.database_url.startswith("sqlite"):
            db_path = self.database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建引擎
        self.engine = create_engine(
            self.database_url,
            echo=settings.app_debug,
            connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
        )
        
        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"数据库连接初始化完成: {self.database_url}")
    
    def init_db(self) -> None:
        """初始化数据库，创建所有表"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("数据库表创建完成")
    
    def drop_all(self) -> None:
        """删除所有表（谨慎使用）"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("所有数据库表已删除")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话的上下文管理器"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()
    
    def get_session_direct(self) -> Session:
        """直接获取会话（需手动管理）"""
        return self.SessionLocal()


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取数据库管理器实例"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.init_db()
    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """FastAPI依赖注入用的数据库会话生成器"""
    db_manager = get_db_manager()
    session = db_manager.SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()



