# 数字人交互助手 - 通义千问版 🤖

一个基于通义千问大语言模型的智能数字人交互系统，支持语音识别、智能对话和语音合成，提供自然流畅的人机交互体验。

## ✨ 特性

- 🎤 **语音识别**: 基于Web Speech API的实时语音转文字
- 🧠 **智能对话**: 集成阿里巴巴通义千问大语言模型，提供智能、自然的对话体验
- 🔊 **语音合成**: 支持多种语音参数调节的文字转语音
- 👁️ **数字人动画**: CSS3驱动的数字人表情和口型同步动画
- 🔒 **HTTPS支持**: 自动生成SSL证书，支持安全的Web Speech API
- 🎛️ **可视化控制**: 直观的用户界面和实时状态反馈
- ⚙️ **灵活配置**: 支持语音参数和AI模型参数调节

## 🛠️ 技术栈

### 前端技术
- **HTML5**: 现代Web标准，语义化页面结构
- **CSS3**: 响应式设计，动画效果，数字人表情动画
- **JavaScript (ES6+)**: 模块化编程，异步处理
- **Web Speech API**: 浏览器原生语音识别和合成
- **Fetch API**: 现代HTTP客户端，与后端API通信

### 后端技术
- **Python 3.7+**: 现代Python特性
- **HTTP服务器**: 基于Python标准库的高性能HTTP服务器
- **SSL/TLS**: 自动证书生成，HTTPS安全连接
- **通义千问API**: 阿里巴巴达摩院大语言模型

### AI模型
- **通义千问-Turbo**: 快速响应，适合日常对话
- **通义千问-Plus**: 平衡性能与质量
- **通义千问-Max**: 最高质量，深度理解和推理

## 📁 项目结构

```
digital_human_assistant/
├── 📄 index.html              # 主页面
├── 🎨 styles.css              # 样式文件
├── ⚙️ app.js                  # 主应用逻辑
├── 🎤 speech-handler.js       # 语音处理模块
├── 😊 digital-human.js        # 数字人动画控制
├── 🐍 server.py               # HTTPS服务器和API代理
├── 📦 requirements.txt        # Python依赖
├── 🪟 start.bat               # Windows启动脚本
├── 🐧 start.sh                # Linux/macOS启动脚本
├── 📁 certificates/           # SSL证书目录
│   ├── 🔒 server.crt          # SSL证书
│   └── 🔑 server.key          # SSL私钥
├── 🔒 server.crt              # SSL证书(根目录备份)
├── 🔑 server.key              # SSL私钥(根目录备份)
└── 📖 README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

确保您的系统已安装以下软件：

- **Python 3.7+** 
- **pip** (Python包管理器)
- **现代浏览器** (Chrome 25+, Firefox 44+, Safari 14.1+, Edge 79+)

### 2. 获取通义千问API密钥

1. 访问 [阿里云控制台](https://dashscope.console.aliyun.com/)
2. 注册并登录您的阿里云账户
3. 开通通义千问服务
4. 创建API密钥
5. 复制您的API密钥备用

### 3. 安装依赖

```bash
# 克隆项目（如果从GitHub获取）
git clone https://github.com/jiankang123/digital_human_assistant.git
cd digital_human_assistant

# 安装Python依赖
pip install -r requirements.txt
```

### 4. 配置API密钥

设置通义千问API密钥环境变量：

**Windows (CMD):**
```cmd
set DASHSCOPE_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:DASHSCOPE_API_KEY="your-api-key-here"
```

**Linux/macOS:**
```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

**永久设置 (Linux/macOS):**
```bash
echo 'export DASHSCOPE_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 5. 启动服务器

**Windows:**
```cmd
# 使用启动脚本
start.bat

# 或直接运行
python server.py
```

**Linux/macOS:**
```bash
# 使用启动脚本
chmod +x start.sh
./start.sh

# 或直接运行
python3 server.py
```

### 6. 访问应用

打开浏览器访问：
- 🌐 **主页**: https://localhost:8444
- 📊 **状态检查**: https://localhost:8444/api/status

**注意**: 首次访问时，浏览器会显示SSL证书警告，点击"高级" → "继续访问localhost"即可。

## 📝 使用说明

### 基础操作

1. **启用语音**: 点击页面上的任何快捷问题按钮以启用音频播放权限
2. **语音交互**: 点击"点击说话"按钮，开始语音对话
3. **文字交互**: 在输入框中输入问题，按回车或点击发送
4. **快捷问题**: 点击预设的快捷问题按钮快速开始对话
5. **停止语音**: 语音播放时点击"停止"按钮可中断播放

### 设置选项

打开设置面板可以调整：

**语音设置**:
- 语音语速 (0.5x - 2.0x)
- 语音音调 (0.0 - 2.0)
- 语音音量 (0.0 - 1.0)
- 语音测试功能

**AI设置**:
- AI模型选择 (Turbo/Plus/Max)
- 创造性控制 (0.0 - 1.0)
- 最大回复长度 (500 - 3000 tokens)

### 语音识别说明

- 🎤 支持中文语音识别
- 🔇 需要HTTPS环境 (已自动配置)
- 📱 支持移动设备浏览器
- 🔊 首次使用需授权麦克风权限

## ⚙️ 配置选项

### 服务器配置

```bash
# 查看所有选项
python server.py --help

# 自定义端口
python server.py --port 9000

# 使用HTTP（禁用SSL）
python server.py --no-ssl

# 自定义SSL证书
python server.py --cert mycert.crt --key mykey.key
```

### AI模型配置

通过修改 `app.js` 中的 `qwenConfig` 对象：

```javascript
this.qwenConfig = {
    apiUrl: '/api/qwen',
    model: 'qwen-turbo',      // qwen-turbo, qwen-plus, qwen-max
    maxTokens: 1500,          // 最大token数
    temperature: 0.7          // 创造性 (0.0-1.0)
};
```

## 🔧 故障排除

### 常见问题

**Q: 无法访问HTTPS链接**
A: 检查SSL证书是否正确生成，或使用 `--no-ssl` 参数启动HTTP服务器

**Q: 语音识别不工作**
A: 确保使用HTTPS访问，并在浏览器中授权麦克风权限

**Q: AI回复显示错误**
A: 检查DASHSCOPE_API_KEY环境变量是否正确设置

**Q: 语音合成不工作**
A: 确保点击了任意按钮以启用音频播放，检查浏览器音频权限

### 调试步骤

1. **检查服务器状态**:
   ```bash
   curl -k https://localhost:8444/api/status
   ```

2. **查看控制台日志**: 
   - 按F12打开浏览器开发者工具
   - 查看Console标签页的错误信息

3. **检查API密钥配置**:
   ```bash
   echo $DASHSCOPE_API_KEY  # Linux/macOS
   echo %DASHSCOPE_API_KEY% # Windows
   ```

4. **测试通义千问API**:
   ```bash
   curl -k -X POST https://localhost:8444/api/qwen \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"你好"}]}'
   ```

### 错误码说明

- **401**: API密钥未设置或无效
- **400**: 请求格式错误
- **500**: 服务器内部错误或通义千问API错误
- **网络错误**: 检查网络连接和API服务状态

## 🚀 高级功能

### 自定义数字人表情

修改 `digital-human.js` 可以自定义数字人的表情和动画：

```javascript
// 添加新表情
setExpression(expression) {
    const expressions = {
        'happy': () => this.showHappyExpression(),
        'sad': () => this.showSadExpression(),
        'custom': () => this.showCustomExpression()  // 自定义表情
    };
}
```

### 扩展语音功能

修改 `speech-handler.js` 可以添加更多语音功能：

```javascript
// 自定义语音识别语言
constructor() {
    this.recognition.lang = 'zh-CN';  // 可更改为其他语言
}
```

### API集成

您可以轻松集成其他AI服务，只需修改 `server.py` 中的API调用部分。

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如果您遇到问题或有建议，请：

1. 查看本文档的故障排除部分
2. 提交 GitHub Issue
3. 查看通义千问API官方文档

---

**享受与您的数字人助手的智能对话吧！** 🎉