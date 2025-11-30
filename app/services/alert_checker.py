"""
价格预警检查器
负责检查价格是否触发预警条件并发送通知
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from loguru import logger

from app.config import settings
from app.database.schemas import AlertConfigDB
from app.services.notifier import EmailNotifier


class AlertChecker:
    """
    价格预警检查器
    
    功能：
    1. 检查价格是否触发预警
    2. 管理冷却时间防止邮件轰炸
    3. 发送邮件通知
    4. 更新预警状态
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.email_notifier = EmailNotifier()
    
    async def check_and_notify(
        self, 
        symbol: str, 
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        检查价格并在需要时发送通知
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            
        Returns:
            触发的预警信息，未触发返回 None
        """
        # 获取该股票的预警配置
        alert_config = self.db.query(AlertConfigDB).filter(
            AlertConfigDB.symbol == symbol,
            AlertConfigDB.is_enabled == True
        ).first()
        
        if not alert_config:
            return None
        
        # 检查是否在冷却期
        if self._is_in_cooldown(alert_config):
            logger.debug(f"{symbol}: 预警在冷却期内，跳过")
            return None
        
        # 检查价格条件
        trigger_result = self._check_trigger_conditions(alert_config, current_price)
        
        if trigger_result:
            # 发送通知
            success = await self._send_alert_notification(
                symbol=symbol,
                current_price=current_price,
                alert_config=alert_config,
                trigger_type=trigger_result['type']
            )
            
            if success:
                # 更新触发时间
                self._update_trigger_time(alert_config, trigger_result['type'])
                
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'trigger_type': trigger_result['type'],
                    'target_price': trigger_result['target'],
                    'timestamp': datetime.now()
                }
        
        return None
    
    def _check_trigger_conditions(
        self, 
        config: AlertConfigDB, 
        price: float
    ) -> Optional[Dict[str, Any]]:
        """
        检查是否触发预警条件
        
        Returns:
            {'type': 'upper'|'lower', 'target': float} 或 None
        """
        # 检查上限
        if config.upper_limit and price >= config.upper_limit:
            logger.info(f"{config.symbol}: 触发上限预警 ({price} >= {config.upper_limit})")
            return {'type': 'upper', 'target': config.upper_limit}
        
        # 检查下限
        if config.lower_limit and price <= config.lower_limit:
            logger.info(f"{config.symbol}: 触发下限预警 ({price} <= {config.lower_limit})")
            return {'type': 'lower', 'target': config.lower_limit}
        
        return None
    
    def _is_in_cooldown(self, config: AlertConfigDB) -> bool:
        """
        检查是否在冷却期内
        
        防止同一预警短时间内重复发送
        """
        if not config.last_triggered_at:
            return False
        
        cooldown_minutes = config.cooldown_minutes or settings.alert_cooldown_minutes
        cooldown_until = config.last_triggered_at + timedelta(minutes=cooldown_minutes)
        
        return datetime.now() < cooldown_until
    
    def _update_trigger_time(self, config: AlertConfigDB, trigger_type: str):
        """更新触发时间"""
        config.last_triggered_at = datetime.now()
        config.last_trigger_type = trigger_type
        self.db.commit()
    
    async def _send_alert_notification(
        self,
        symbol: str,
        current_price: float,
        alert_config: AlertConfigDB,
        trigger_type: str
    ) -> bool:
        """
        发送预警通知
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            alert_config: 预警配置
            trigger_type: 触发类型 'upper' | 'lower'
        """
        target_price = (
            alert_config.upper_limit if trigger_type == 'upper' 
            else alert_config.lower_limit
        )
        
        # 发送邮件
        if self.email_notifier.is_configured():
            recipient = alert_config.email or settings.email_default_recipient
            
            if recipient:
                success = await self.email_notifier.send_price_alert(
                    symbol=symbol,
                    current_price=current_price,
                    target_price=target_price,
                    alert_type=trigger_type,
                    recipient=recipient
                )
                return success
            else:
                logger.warning(f"{symbol}: 预警触发但未配置收件人")
        else:
            logger.warning("邮件未配置，无法发送预警通知")
        
        return False
    
    # ==================== 预警配置管理 ====================
    
    def get_alert_config(self, symbol: str) -> Optional[AlertConfigDB]:
        """获取股票的预警配置"""
        return self.db.query(AlertConfigDB).filter(
            AlertConfigDB.symbol == symbol
        ).first()
    
    def set_alert_config(
        self,
        symbol: str,
        upper_limit: Optional[float] = None,
        lower_limit: Optional[float] = None,
        email: Optional[str] = None,
        cooldown_minutes: int = 30
    ) -> AlertConfigDB:
        """
        设置或更新预警配置
        
        Args:
            symbol: 股票代码
            upper_limit: 价格上限
            lower_limit: 价格下限
            email: 接收邮箱
            cooldown_minutes: 冷却时间（分钟）
        """
        config = self.get_alert_config(symbol)
        
        if config:
            # 更新现有配置
            if upper_limit is not None:
                config.upper_limit = upper_limit
            if lower_limit is not None:
                config.lower_limit = lower_limit
            if email is not None:
                config.email = email
            config.cooldown_minutes = cooldown_minutes
            config.updated_at = datetime.now()
        else:
            # 创建新配置
            config = AlertConfigDB(
                symbol=symbol,
                upper_limit=upper_limit,
                lower_limit=lower_limit,
                email=email,
                is_enabled=True,
                cooldown_minutes=cooldown_minutes
            )
            self.db.add(config)
        
        self.db.commit()
        logger.info(f"设置 {symbol} 预警: 上限={upper_limit}, 下限={lower_limit}")
        return config
    
    def enable_alert(self, symbol: str) -> bool:
        """启用预警"""
        config = self.get_alert_config(symbol)
        if config:
            config.is_enabled = True
            self.db.commit()
            return True
        return False
    
    def disable_alert(self, symbol: str) -> bool:
        """禁用预警"""
        config = self.get_alert_config(symbol)
        if config:
            config.is_enabled = False
            self.db.commit()
            return True
        return False
    
    def delete_alert(self, symbol: str) -> bool:
        """删除预警配置"""
        deleted = self.db.query(AlertConfigDB).filter(
            AlertConfigDB.symbol == symbol
        ).delete()
        self.db.commit()
        return deleted > 0
    
    def get_all_enabled_alerts(self) -> List[AlertConfigDB]:
        """获取所有启用的预警配置"""
        return self.db.query(AlertConfigDB).filter(
            AlertConfigDB.is_enabled == True
        ).all()
    
    def get_alert_status(self, symbol: str) -> Dict[str, Any]:
        """
        获取预警状态摘要
        
        Returns:
            {
                'configured': bool,
                'enabled': bool,
                'upper_limit': float,
                'lower_limit': float,
                'last_triggered': datetime,
                'in_cooldown': bool
            }
        """
        config = self.get_alert_config(symbol)
        
        if not config:
            return {
                'configured': False,
                'enabled': False,
                'upper_limit': None,
                'lower_limit': None,
                'last_triggered': None,
                'in_cooldown': False
            }
        
        return {
            'configured': True,
            'enabled': config.is_enabled,
            'upper_limit': config.upper_limit,
            'lower_limit': config.lower_limit,
            'email': config.email,
            'last_triggered': config.last_triggered_at,
            'last_trigger_type': config.last_trigger_type,
            'in_cooldown': self._is_in_cooldown(config),
            'cooldown_minutes': config.cooldown_minutes
        }


# 便捷函数
def create_alert_checker(db_session: Session) -> AlertChecker:
    """创建预警检查器实例"""
    return AlertChecker(db_session)

