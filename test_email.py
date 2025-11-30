"""
邮件发送测试脚本
用于验证邮件配置是否正确
"""

import asyncio
import sys
from app.services.notifier import EmailNotifier
from app.config import settings

async def test_email(recipient_email: str):
    """发送测试邮件"""
    
    print("=" * 50)
    print("AITrading 邮件发送测试")
    print("=" * 50)
    print()
    
    # 检查配置
    notifier = EmailNotifier()
    
    print(f"SMTP服务器: {notifier.smtp_host}:{notifier.smtp_port}")
    print(f"发送方邮箱: {notifier.sender or '(未配置)'}")
    print(f"是否已配置: {notifier.is_configured()}")
    print()
    
    if not notifier.is_configured():
        print("❌ 错误: 邮件未配置!")
        print()
        print("请在 .env 文件中配置:")
        print("  EMAIL_SENDER=your_qq@qq.com")
        print("  EMAIL_PASSWORD=your_authorization_code")
        print()
        print("获取授权码步骤:")
        print("  1. 登录 QQ 邮箱")
        print("  2. 设置 -> 账户")
        print("  3. 开启 POP3/SMTP 服务")
        print("  4. 生成授权码")
        return False
    
    print(f"收件人: {recipient_email}")
    print()
    print("正在发送测试邮件...")
    print()
    
    # 发送测试邮件
    success = await notifier.send_test_email(recipient=recipient_email)
    
    if success:
        print("✅ 邮件发送成功!")
        print()
        print(f"请检查 {recipient_email} 的收件箱（包括垃圾邮件文件夹）")
        return True
    else:
        print("❌ 邮件发送失败!")
        print()
        print("可能的原因:")
        print("  1. 授权码错误")
        print("  2. SMTP服务器连接失败")
        print("  3. 收件人邮箱地址无效")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        recipient = sys.argv[1]
    else:
        recipient = input("请输入接收测试邮件的邮箱地址: ").strip()
    
    if not recipient:
        print("❌ 错误: 未提供邮箱地址")
        sys.exit(1)
    
    result = asyncio.run(test_email(recipient))
    sys.exit(0 if result else 1)

