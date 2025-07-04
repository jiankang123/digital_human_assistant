#!/bin/bash

# 数字人交互助手 - 通义千问版
# Linux/macOS 启动脚本
# 版本: 2.0.0

echo "🤖 数字人交互助手 - 通义千问版"
echo "===================================="

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    echo "请安装 Python 3.7+ 后重试"
    exit 1
fi

# 检查API密钥
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "⚠️  警告: 未设置 DASHSCOPE_API_KEY 环境变量"
    echo "请设置您的通义千问API密钥:"
    echo "export DASHSCOPE_API_KEY=\"your-api-key-here\""
    echo ""
    read -p "是否继续启动 (可能会有API错误)? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消启动"
        exit 1
    fi
fi

# 检查依赖
echo "📦 检查Python依赖..."
python3 -c "import ssl, json, urllib.request, urllib.parse" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  正在安装依赖..."
    pip3 install -r requirements.txt
fi

# 启动服务器
echo "🚀 启动HTTPS服务器..."
echo "📍 服务地址: https://localhost:8444"
echo "📊 状态检查: https://localhost:8444/api/status"
echo ""
echo "💡 提示: 首次访问时，浏览器会显示SSL证书警告，"
echo "   点击 '高级' → '继续访问localhost' 即可。"
echo ""
echo "⏹️  按 Ctrl+C 停止服务器"
echo ""

python3 server.py