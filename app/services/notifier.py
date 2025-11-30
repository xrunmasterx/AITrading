"""
通知服务模块（预留接口）
支持多种通知方式：邮件、Telegram、微信等
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger

from app.config import settings


class NotificationChannel(ABC):
    """通知渠道基类"""
    
    @abstractmethod
    async def send(self, message: str, **kwargs) -> bool:
        """发送通知"""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """检查是否已配置"""
        pass


class TelegramNotifier(NotificationChannel):
    """Telegram通知器"""
    
    def __init__(self):
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
    
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)
    
    async def send(self, message: str, **kwargs) -> bool:
        """
        发送Telegram消息
        
        Args:
            message: 消息内容
            parse_mode: 解析模式 (HTML/Markdown)
        """
        if not self.is_configured():
            logger.warning("Telegram未配置")
            return False
        
        try:
            import httpx
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": kwargs.get("parse_mode", "HTML")
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data)
                
                if response.status_code == 200:
                    logger.info("Telegram消息发送成功")
                    return True
                else:
                    logger.error(f"Telegram发送失败: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Telegram发送异常: {e}")
            return False


class EmailNotifier(NotificationChannel):
    """
    邮件通知器（QQ邮箱 SMTP）
    
    使用说明：
    1. 登录 QQ 邮箱
    2. 设置 -> 账户 -> POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务
    3. 开启 SMTP 服务，获取授权码
    4. 在 .env 中配置 EMAIL_SENDER 和 EMAIL_PASSWORD
    """
    
    def __init__(self):
        self.smtp_host = settings.email_smtp_host
        self.smtp_port = settings.email_smtp_port
        self.sender = settings.email_sender
        self.password = settings.email_password
        self.default_recipient = settings.email_default_recipient
    
    def is_configured(self) -> bool:
        """检查是否已配置邮件"""
        return bool(self.sender and self.password)
    
    async def send(self, message: str, **kwargs) -> bool:
        """
        发送邮件
        
        Args:
            message: 邮件正文
            subject: 邮件主题（可选）
            recipient: 收件人（可选，默认使用配置的收件人）
            html: 是否使用HTML格式（可选）
        """
        if not self.is_configured():
            logger.warning("邮件未配置: 请在 .env 中设置 EMAIL_SENDER 和 EMAIL_PASSWORD")
            return False
        
        recipient = kwargs.get('recipient') or self.default_recipient
        if not recipient:
            logger.warning("邮件未配置收件人")
            return False
        
        subject = kwargs.get('subject', 'AITrading 通知')
        is_html = kwargs.get('html', False)
        
        return await self._send_email(
            recipient=recipient,
            subject=subject,
            body=message,
            is_html=is_html
        )
    
    async def _send_email(
        self, 
        recipient: str, 
        subject: str, 
        body: str,
        is_html: bool = False
    ) -> bool:
        """
        发送邮件（内部方法）
        
        使用 smtplib 同步发送（在线程池中执行）
        """
        import asyncio
        import smtplib
        import ssl
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from email.header import Header
        
        def _sync_send():
            try:
                # 创建邮件对象
                msg = MIMEMultipart('alternative')
                msg['From'] = self.sender
                msg['To'] = recipient
                msg['Subject'] = Header(subject, 'utf-8')
                
                # 添加正文
                content_type = 'html' if is_html else 'plain'
                msg.attach(MIMEText(body, content_type, 'utf-8'))
                
                # 创建 SSL 上下文
                context = ssl.create_default_context()
                
                # 连接 SMTP 服务器并发送
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, [recipient], msg.as_string())
                
                logger.info(f"邮件发送成功: {recipient}")
                return True
                
            except smtplib.SMTPAuthenticationError as e:
                error_msg = f"邮件认证失败（请检查授权码是否正确）: {e}"
                logger.error(error_msg)
                print(f"详细错误: {type(e).__name__}: {e}")
                return False
            except smtplib.SMTPException as e:
                error_msg = f"邮件发送失败: {e}"
                logger.error(error_msg)
                print(f"详细错误: {type(e).__name__}: {e}")
                print(f"错误代码: {e.smtp_code if hasattr(e, 'smtp_code') else 'N/A'}")
                print(f"错误消息: {e.smtp_error if hasattr(e, 'smtp_error') else 'N/A'}")
                return False
            except Exception as e:
                error_msg = f"邮件发送异常: {e}"
                logger.error(error_msg)
                print(f"详细错误: {type(e).__name__}: {e}")
                import traceback
                print(traceback.format_exc())
                return False
        
        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_send)
    
    async def send_price_alert(
        self,
        symbol: str,
        current_price: float,
        target_price: float,
        alert_type: str,
        recipient: Optional[str] = None
    ) -> bool:
        """
        发送价格预警邮件
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            target_price: 目标价格
            alert_type: 'upper' 或 'lower'
            recipient: 收件人（可选）
        """
        direction = "突破上限" if alert_type == "upper" else "跌破下限"
        emoji = "📈" if alert_type == "upper" else "📉"
        
        subject = f"[AITrading] {emoji} {symbol} 价格预警"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: {'#00C853' if alert_type == 'upper' else '#FF5252'};">
                {emoji} 价格预警
            </h2>
            <table style="border-collapse: collapse; margin: 20px 0;">
                <tr>
                    <td style="padding: 8px; font-weight: bold;">股票代码：</td>
                    <td style="padding: 8px;">{symbol}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">当前价格：</td>
                    <td style="padding: 8px; font-size: 1.2em; color: {'#00C853' if alert_type == 'upper' else '#FF5252'};">
                        ${current_price:.2f}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">触发条件：</td>
                    <td style="padding: 8px;">{direction} ${target_price:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">触发时间：</td>
                    <td style="padding: 8px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>
            <p style="color: #666; font-size: 0.9em;">
                ⚠️ 本通知由 AITrading 自动发送，数据可能存在15分钟延迟，仅供参考。
            </p>
        </body>
        </html>
        """
        
        return await self.send(
            message=body,
            subject=subject,
            recipient=recipient,
            html=True
        )
    
    async def send_test_email(self, recipient: Optional[str] = None) -> bool:
        """
        发送测试邮件
        
        用于验证邮件配置是否正确
        """
        subject = "[AITrading] 测试邮件"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #2196F3;">✅ 邮件配置成功！</h2>
            <p>您的 AITrading 邮件通知已配置成功。</p>
            <p>发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <hr>
            <p style="color: #666; font-size: 0.9em;">
                此邮件由 AITrading 量化分析系统发送。
            </p>
        </body>
        </html>
        """
        
        return await self.send(
            message=body,
            subject=subject,
            recipient=recipient,
            html=True
        )


class WeChatNotifier(NotificationChannel):
    """微信通知器（预留，通过企业微信或Server酱）"""
    
    def __init__(self):
        self.webhook_url = ""
    
    def is_configured(self) -> bool:
        return False  # 暂未实现
    
    async def send(self, message: str, **kwargs) -> bool:
        """发送微信消息"""
        if not self.is_configured():
            logger.warning("微信通知未配置")
            return False
        
        # TODO: 实现微信通知
        logger.info("微信发送功能待实现")
        return False


class Notifier:
    """统一通知管理器"""
    
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {
            "telegram": TelegramNotifier(),
            "email": EmailNotifier(),
            "wechat": WeChatNotifier()
        }
    
    def get_available_channels(self) -> List[str]:
        """获取已配置的通知渠道"""
        return [
            name for name, channel in self.channels.items()
            if channel.is_configured()
        ]
    
    async def send(
        self, 
        message: str, 
        channels: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, bool]:
        """
        发送通知到指定渠道
        
        Args:
            message: 消息内容
            channels: 目标渠道列表，None表示所有已配置渠道
            
        Returns:
            各渠道发送结果
        """
        if channels is None:
            channels = self.get_available_channels()
        
        results = {}
        for channel_name in channels:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                if channel.is_configured():
                    results[channel_name] = await channel.send(message, **kwargs)
                else:
                    results[channel_name] = False
                    logger.warning(f"通知渠道 {channel_name} 未配置")
        
        return results
    
    async def send_price_alert(
        self,
        symbol: str,
        current_price: float,
        target_price: float,
        alert_type: str = "above"
    ) -> Dict[str, bool]:
        """
        发送价格预警通知
        
        Args:
            symbol: 股票代码
            current_price: 当前价格
            target_price: 目标价格
            alert_type: 预警类型 (above/below)
        """
        direction = "突破" if alert_type == "above" else "跌破"
        
        message = (
            f"🔔 <b>价格预警</b>\n\n"
            f"股票: <b>{symbol}</b>\n"
            f"当前价格: ${current_price:.2f}\n"
            f"触发条件: {direction} ${target_price:.2f}\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send(message)
    
    async def send_analysis_report(
        self,
        symbol: str,
        summary: str
    ) -> Dict[str, bool]:
        """
        发送分析报告通知
        
        Args:
            symbol: 股票代码
            summary: 分析摘要
        """
        message = (
            f"📊 <b>分析报告 - {symbol}</b>\n\n"
            f"{summary}\n\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send(message)
    
    async def send_daily_summary(
        self,
        symbol: str,
        price: float,
        change_percent: float,
        sentiment: str
    ) -> Dict[str, bool]:
        """
        发送每日摘要
        
        Args:
            symbol: 股票代码
            price: 收盘价
            change_percent: 涨跌幅
            sentiment: 舆情状态
        """
        emoji = "📈" if change_percent > 0 else "📉" if change_percent < 0 else "➖"
        
        message = (
            f"📅 <b>每日摘要 - {symbol}</b>\n\n"
            f"{emoji} 价格: ${price:.2f} ({change_percent:+.2f}%)\n"
            f"📰 舆情: {sentiment}\n"
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        return await self.send(message)


# 创建全局实例
def create_notifier() -> Notifier:
    """创建通知管理器实例"""
    return Notifier()



