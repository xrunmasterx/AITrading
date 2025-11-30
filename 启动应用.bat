@echo off
chcp 65001 >nul
title AITrading - 量化分析工具
color 0A

echo ========================================
echo    AITrading 量化分析工具
echo ========================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 检查 Python 是否安装
echo 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    pause
    exit /b 1
)

REM 检查虚拟环境
if exist "venv\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
)

echo.
echo 正在启动应用...
echo 浏览器将自动打开 http://localhost:8501
echo 按 Ctrl+C 可停止应用
echo.

REM 启动 Streamlit
python -m streamlit run app/ui.py --server.port 8501

if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查：
    echo 1. Python 是否正确安装
    echo 2. 依赖是否已安装 (pip install -r requirements.txt)
    echo 3. 端口 8501 是否被占用
    echo.
    pause
)

