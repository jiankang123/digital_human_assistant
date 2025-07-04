/**
 * 数字人交互助手主应用程序 - 通义千问版本
 * @author Assistant
 * @version 2.0.0
 */

class DigitalHumanApp {
    constructor() {
        this.initialized = false;
        this.isListening = false;
        this.isSpeaking = false;
        
        // 服务实例
        this.speechHandler = null;
        this.digitalHuman = null;
        
        // UI元素
        this.elements = {};
        
        // 语音设置
        this.voiceSettings = {
            rate: 1.0,
            pitch: 1.0,
            volume: 1.0
        };

        // 通义千问配置
        this.qwenConfig = {
            apiUrl: '/api/qwen',  // 使用本地代理
            model: 'qwen-turbo',
            maxTokens: 1500,
            temperature: 0.7
        };

        // 对话历史（用于上下文）
        this.conversationHistory = [];
    }

    /**
     * 初始化应用程序
     */
    async init() {
        try {
            console.log('正在初始化数字人交互助手...');
            
            // 获取UI元素
            this.initializeElements();
            
            // 初始化服务
            await this.initializeServices();
            
            // 绑定事件监听器
            this.bindEventListeners();
            
            // 更新UI状态
            this.updateUIState();
            
            this.initialized = true;
            console.log('数字人助手初始化完成');
            
            // 添加全局调试功能
            window.debugTTS = () => {
                if (this.speechHandler) {
                    const status = this.speechHandler.getTTSStatus();
                    console.table(status);
                    return status;
                }
                return null;
            };
            
            window.resetTTS = () => {
                return this.resetTTS();
            };
            
            // 欢迎消息
            this.addMessage('system', '您好！我是基于通义千问的数字人助手，很高兴为您服务！');
            
        } catch (error) {
            console.error('初始化失败:', error);
            this.showError('初始化失败，请刷新页面重试');
        }
    }

    /**
     * 初始化UI元素
     */
    initializeElements() {
        this.elements = {
            voiceBtn: document.getElementById('voiceBtn'),
            stopBtn: document.getElementById('stopBtn'),
            textInput: document.getElementById('textInput'),
            chatMessages: document.getElementById('chatMessages'),
            settingsPanel: document.getElementById('settingsPanel')
        };

        // 检查必要元素
        const requiredElements = ['voiceBtn', 'textInput', 'chatMessages'];
        for (const elementId of requiredElements) {
            if (!this.elements[elementId]) {
                throw new Error(`必需的UI元素未找到: ${elementId}`);
            }
        }
    }

    /**
     * 初始化服务
     */
    async initializeServices() {
        try {
            // 初始化语音处理器
            this.speechHandler = new SpeechHandler();

            // 设置语音处理器回调
            this.speechHandler.setCallback('onSpeechStart', () => {
                this.isListening = true;
                this.updateVoiceButtonState();
                this.digitalHuman?.startListening();
            });

            this.speechHandler.setCallback('onSpeechEnd', () => {
                this.isListening = false;
                this.updateVoiceButtonState();
                this.digitalHuman?.stopListening();
            });

            this.speechHandler.setCallback('onSpeechResult', (result) => {
                if (result.final) {
                    this.handleSpeechResult(result.final);
                }
            });

            this.speechHandler.setCallback('onSpeechError', (error) => {
                console.error('语音识别错误:', error);
                this.showError(error);
            });

            this.speechHandler.setCallback('onTTSStart', (text) => {
                this.isSpeaking = true;
                this.updateStopButtonState();
                this.digitalHuman?.startTalking();
                console.log('开始语音播放:', text);
            });

            this.speechHandler.setCallback('onTTSEnd', () => {
                this.isSpeaking = false;
                this.updateStopButtonState();
                this.digitalHuman?.stopTalking();
                console.log('语音播放结束');
            });

            this.speechHandler.setCallback('onTTSError', (error) => {
                console.error('语音合成错误:', error);
                this.isSpeaking = false;
                this.updateStopButtonState();
                this.digitalHuman?.setEmotion('neutral');
                
                // 过滤掉interrupted错误（用户主动停止）
                if (error.includes('interrupted')) {
                    console.log('用户主动停止语音播放');
                    return;
                }
                
                this.showError('语音播放失败: ' + error);
            });

            // 初始化数字人
            this.digitalHuman = new DigitalHuman();
            this.digitalHuman.init();

        } catch (error) {
            console.error('服务初始化失败:', error);
            throw error;
        }
    }

    /**
     * 绑定事件监听器
     */
    bindEventListeners() {
        // 语音按钮
        this.elements.voiceBtn?.addEventListener('click', () => this.toggleVoice());
        
        // 停止按钮
        this.elements.stopBtn?.addEventListener('click', () => this.stopSpeaking());
        
        // 文本输入
        this.elements.textInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextMessage();
            }
        });

        // 设置按钮
        document.getElementById('settingsBtn')?.addEventListener('click', () => {
            window.toggleSettings();
        });

        // 关闭设置按钮
        document.getElementById('closeSettings')?.addEventListener('click', () => {
            const panel = document.getElementById('settingsPanel');
            if (panel) {
                panel.classList.remove('show');
            }
        });

        // 清空聊天记录按钮
        document.getElementById('clearHistoryBtn')?.addEventListener('click', () => {
            this.clearChatHistory();
        });

        // 快捷问题按钮
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const question = e.target.textContent.trim();
                this.askQuestion(question);
            });
        });

        // 语音设置
        document.getElementById('speechSpeed')?.addEventListener('change', (e) => {
            this.voiceSettings.rate = parseFloat(e.target.value);
            document.getElementById('speedValue').textContent = e.target.value;
            this.updateVoiceSettings();
        });

        document.getElementById('speechPitch')?.addEventListener('change', (e) => {
            this.voiceSettings.pitch = parseFloat(e.target.value);
            document.getElementById('pitchValue').textContent = e.target.value;
            this.updateVoiceSettings();
        });

        document.getElementById('speechVolume')?.addEventListener('change', (e) => {
            this.voiceSettings.volume = parseFloat(e.target.value);
            document.getElementById('volumeValue').textContent = e.target.value;
            this.updateVoiceSettings();
        });

        // 测试语音按钮
        document.getElementById('testVoiceBtn')?.addEventListener('click', () => {
            this.testVoice();
        });

        // TTS重置按钮
        document.getElementById('resetTtsBtn')?.addEventListener('click', async () => {
            await this.resetTTS();
        });

        // 定期更新TTS状态
        setInterval(() => {
            this.updateTTSStatus();
        }, 2000);
    }

    /**
     * 切换语音识别
     */
    toggleVoice() {
        if (!this.initialized) return;

        if (this.isListening) {
            this.speechHandler?.stopListening();
        } else {
            this.speechHandler?.startListening();
        }
    }

    /**
     * 停止语音播放
     */
    stopSpeaking() {
        if (this.isSpeaking) {
            this.speechHandler?.stopSpeaking();
        }
    }

    /**
     * 发送文本消息
     */
    sendTextMessage() {
        const textInput = this.elements.textInput;
        const message = textInput.value.trim();
        
        if (message) {
            textInput.value = '';
            this.processUserMessage(message);
        }
    }

    /**
     * 询问问题
     */
    askQuestion(question) {
        this.processUserMessage(question);
    }

    /**
     * 处理语音识别结果
     */
    handleSpeechResult(transcript) {
        console.log('语音识别结果:', transcript);
        this.processUserMessage(transcript);
    }

    /**
     * 处理用户消息 - 调用通义千问API
     */
    async processUserMessage(message) {
        try {
            // 添加用户消息到聊天历史
            this.addMessage('user', message);

            // 显示思考状态
            this.digitalHuman?.showThinking();

            // 调用通义千问API获取回答
            const response = await this.callQwenAPI(message);

            // 添加助手回答到聊天历史
            this.addMessage('assistant', response);

            // 语音播放回答
            if (this.speechHandler) {
                try {
                    await this.speechHandler.speak(response);
                } catch (speechError) {
                    console.warn('语音播放失败:', speechError.message);
                    // 语音播放失败不影响文本回答的显示
                }
            }

            // 重置表情
            this.digitalHuman?.setEmotion('neutral');

        } catch (error) {
            console.error('处理消息失败:', error);
            console.error('错误详情:', error.stack);
            
            // 重置表情
            this.digitalHuman?.setEmotion('neutral');
            
            // 根据错误类型显示不同的错误消息
            let errorMessage = '抱歉，我遇到了一些问题，请稍后再试。';
            
            if (error.message.includes('API请求失败')) {
                errorMessage = '网络连接出现问题，请检查网络后重试。';
            } else if (error.message.includes('API返回错误')) {
                errorMessage = 'AI服务暂时不可用，请稍后再试。';
            } else if (error.message.includes('获取AI回答失败')) {
                errorMessage = '获取回答失败，请重新提问。';
            }
            
            this.addMessage('assistant', errorMessage);
            
            // 尝试语音播放错误消息
            if (this.speechHandler) {
                try {
                    await this.speechHandler.speak(errorMessage);
                } catch (speechError) {
                    console.warn('错误消息语音播放失败:', speechError.message);
                }
            }
        }
    }

    /**
     * 调用通义千问API
     */
    async callQwenAPI(message) {
        try {
            // 添加当前消息到对话历史
            this.conversationHistory.push({
                role: 'user',
                content: message
            });

            // 保持对话历史在合理长度内（最近10轮对话）
            if (this.conversationHistory.length > 20) {
                this.conversationHistory = this.conversationHistory.slice(-20);
            }

            const requestBody = {
                model: this.qwenConfig.model,
                messages: [
                    {
                        role: 'system',
                        content: '你是一个友善、专业的AI助手。请用简洁、自然的中文回答用户的问题。回答要准确、有帮助，并且适合语音播放。'
                    },
                    ...this.conversationHistory
                ],
                max_tokens: this.qwenConfig.maxTokens,
                temperature: this.qwenConfig.temperature
            };

            console.log('发送API请求:', requestBody);

            const response = await fetch(this.qwenConfig.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            console.log('API响应状态:', response.status, response.statusText);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('API错误响应:', errorText);
                throw new Error(`API请求失败: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            console.log('API响应数据:', data);
            
            if (data.error) {
                console.error('API返回错误:', data.error);
                throw new Error(data.error.message || 'API返回错误');
            }

            const assistantMessage = data.choices?.[0]?.message?.content;
            if (!assistantMessage) {
                console.error('API响应格式错误:', data);
                throw new Error('API返回的响应格式不正确');
            }

            // 添加助手回答到对话历史
            this.conversationHistory.push({
                role: 'assistant',
                content: assistantMessage
            });

            console.log('API调用成功，回答:', assistantMessage);
            return assistantMessage;

        } catch (error) {
            console.error('调用通义千问API失败:', error);
            console.error('错误详情:', error.stack);
            throw new Error('获取AI回答失败: ' + error.message);
        }
    }

    /**
     * 添加消息到聊天历史
     */
    addMessage(type, content) {
        const messagesContainer = this.elements.chatMessages;
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = type === 'user' ? '👤' : '🤖';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = new Date().toLocaleTimeString();
        contentDiv.appendChild(timeSpan);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    /**
     * 清除聊天历史
     */
    clearChatHistory() {
        const messagesContainer = this.elements.chatMessages;
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }
        // 同时清除对话历史
        this.conversationHistory = [];
    }

    /**
     * 更新语音按钮状态
     */
    updateVoiceButtonState() {
        const voiceBtn = this.elements.voiceBtn;
        if (!voiceBtn) return;

        if (this.isListening) {
            voiceBtn.classList.add('listening');
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i><span class="btn-text">停止录音</span>';
        } else {
            voiceBtn.classList.remove('listening');
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i><span class="btn-text">开始语音</span>';
        }
    }

    /**
     * 更新停止按钮状态
     */
    updateStopButtonState() {
        const stopBtn = this.elements.stopBtn;
        if (stopBtn) {
            stopBtn.disabled = !this.isSpeaking;
        }
    }

    /**
     * 更新UI状态
     */
    updateUIState() {
        this.updateVoiceButtonState();
        this.updateStopButtonState();
    }

    /**
     * 显示错误信息
     */
    showError(message) {
        console.error('错误:', message);
        this.addMessage('system', `错误: ${message}`);
    }

    /**
     * 显示成功信息
     */
    showSuccess(message) {
        console.log('成功:', message);
        this.addMessage('system', `成功: ${message}`);
    }

    /**
     * 测试语音
     */
    testVoice() {
        if (this.speechHandler) {
            this.speechHandler.speak('这是语音测试，您能听到我的声音吗？');
        }
    }

    /**
     * 更新语音设置
     */
    updateVoiceSettings() {
        if (this.speechHandler) {
            this.speechHandler.updateConfig({
                synthesis: this.voiceSettings
            });
        }
    }

    /**
     * 重置TTS引擎
     */
    async resetTTS() {
        const resetBtn = document.getElementById('resetTtsBtn');
        const statusText = document.getElementById('statusText');
        
        if (resetBtn) {
            resetBtn.classList.add('resetting');
            resetBtn.disabled = true;
        }
        
        if (statusText) {
            statusText.textContent = '重置中...';
        }
        
        try {
            if (this.speechHandler) {
                await this.speechHandler.manualResetTTS();
                this.showSuccess('TTS引擎重置成功');
            }
        } catch (error) {
            console.error('TTS重置失败:', error);
            this.showError('TTS重置失败: ' + error.message);
        } finally {
            if (resetBtn) {
                resetBtn.classList.remove('resetting');
                resetBtn.disabled = false;
            }
            
            // 更新状态
            setTimeout(() => {
                this.updateTTSStatus();
            }, 500);
        }
    }

    /**
     * 更新TTS状态显示
     */
    updateTTSStatus() {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (!statusIndicator || !statusText || !this.speechHandler) {
            return;
        }
        
        try {
            const status = this.speechHandler.getTTSStatus();
            
            // 移除之前的状态类
            statusIndicator.classList.remove('warning', 'error');
            
            if (status.queueLength > 3) {
                // 队列过长
                statusIndicator.classList.add('warning');
                statusText.textContent = `队列繁忙 (${status.queueLength})`;
            } else if (status.synthesisPending && !status.isSpeaking) {
                // 可能卡住了
                statusIndicator.classList.add('error');
                statusText.textContent = '可能卡住';
            } else if (status.isSpeaking) {
                // 正在播放
                statusText.textContent = '正在播放';
            } else if (status.queueLength > 0) {
                // 有队列
                statusText.textContent = `队列: ${status.queueLength}`;
            } else {
                // 正常状态
                statusText.textContent = '就绪';
            }
            
            // 显示语音名称
            if (status.voiceName && status.voiceName !== 'default') {
                statusText.title = `语音: ${status.voiceName}`;
            }
            
        } catch (error) {
            statusIndicator.classList.add('error');
            statusText.textContent = '状态异常';
            console.error('获取TTS状态失败:', error);
        }
    }
}

// 全局函数
window.toggleSettings = function() {
    const panel = document.getElementById('settingsPanel');
    if (panel) {
        panel.classList.toggle('show');
    }
};

window.showTab = function(tabName) {
    // 隐藏所有标签内容
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // 移除所有标签按钮的活动状态
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 显示指定标签内容
    const targetContent = document.getElementById(tabName + 'Tab');
    if (targetContent) {
        targetContent.classList.add('active');
    }
    
    // 激活对应的标签按钮
    event.target.classList.add('active');
};

window.toggleVoice = function() {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.toggleVoice();
    }
};

window.stopSpeaking = function() {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.stopSpeaking();
    }
};

window.sendTextMessage = function() {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.sendTextMessage();
    }
};

window.askQuestion = function(question) {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.askQuestion(question);
    }
};

window.clearChatHistory = function() {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.clearChatHistory();
    }
};

window.testVoice = function() {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.testVoice();
    }
};

window.updateVoiceSettings = function() {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.updateVoiceSettings();
    }
};

// 初始化应用程序
document.addEventListener('DOMContentLoaded', () => {
    window.digitalHumanApp = new DigitalHumanApp();
    window.digitalHumanApp.init();
});

// 全局错误处理
window.addEventListener('error', (event) => {
    console.error('全局错误:', event.error);
    console.error('错误位置:', event.filename, '行:', event.lineno, '列:', event.colno);
    console.error('错误堆栈:', event.error?.stack);
    
    // 过滤掉一些已知的非关键错误
    const errorMessage = event.error?.message || event.message || '';
    
    // 跳过语音相关的非关键错误
    if (errorMessage.includes('语音') || 
        errorMessage.includes('speech') ||
        errorMessage.includes('audio') ||
        errorMessage.includes('microphone') ||
        event.filename?.includes('speech-handler.js')) {
        console.warn('跳过语音相关错误:', errorMessage);
        return;
    }
    
    // 跳过网络请求错误（这些应该由API调用处理）
    if (errorMessage.includes('fetch') || 
        errorMessage.includes('network') ||
        errorMessage.includes('Failed to fetch')) {
        console.warn('跳过网络错误:', errorMessage);
        return;
    }
    
    // 只处理真正的JavaScript错误
    if (window.digitalHumanApp && event.error instanceof Error) {
        window.digitalHumanApp.showError('JavaScript错误: ' + errorMessage);
    }
});

// 处理未捕获的Promise错误
window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise错误:', event.reason);
    
    // 阻止默认的错误处理
    event.preventDefault();
    
    // 如果是语音相关错误，不显示给用户
    const reason = event.reason?.message || event.reason || '';
    if (typeof reason === 'string' && 
        (reason.includes('语音') || 
         reason.includes('speech') ||
         reason.includes('audio') ||
         reason.includes('需要用户交互'))) {
        console.warn('跳过语音Promise错误:', reason);
        return;
    }
    
    // 其他Promise错误才显示给用户
    if (window.digitalHumanApp) {
        window.digitalHumanApp.showError('系统错误，请重试');
    }
});

// 页面卸载时清理资源
window.addEventListener('beforeunload', () => {
    if (window.digitalHumanApp) {
        window.digitalHumanApp.speechHandler?.destroy();
        window.digitalHumanApp.digitalHuman?.destroy();
    }
});