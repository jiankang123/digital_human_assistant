@echo off
chcp 65001 >nul
title 数字人交互助手 - 通义千问版

echo 🤖 数字人交互助手 - 通义千问版
echo ====================================

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到 Python
    echo 请安装 Python 3.7+ 后重试
    echo.
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查API密钥
if "%DASHSCOPE_API_KEY%"=="" (
    echo ⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量
    echo 请设置您的通义千问API密钥:
    echo set DASHSCOPE_API_KEY=your-api-key-here
    echo.
    set /p continue="是否继续启动 (可能会有API错误)? [y/N]: "
    if /i not "%continue%"=="y" (
        echo 已取消启动
        pause
        exit /b 1
    )
)

REM 检查依赖
echo 📦 检查Python依赖...
python -c "import ssl, json, urllib.request, urllib.parse" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  正在安装依赖...
    pip install -r requirements.txt
)

REM 启动服务器
echo 🚀 启动HTTPS服务器...
echo 📍 服务地址: https://localhost:8444
echo 📊 状态检查: https://localhost:8444/api/status
echo.
echo 💡 提示: 首次访问时，浏览器会显示SSL证书警告，
echo    点击 "高级" → "继续访问localhost" 即可。
echo.
echo ⏹️  按 Ctrl+C 停止服务器
echo.

python server.py

echo.
echo 👋 服务器已停止
pause