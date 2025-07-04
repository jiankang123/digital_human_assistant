// 数字人动画控制器
class DigitalHuman {
    constructor() {
        this.avatarElement = null;
        this.mouthElement = null;
        this.eyeElements = null;
        this.statusElement = null;
        this.visualizerElement = null;
        
        // 动画状态
        this.animationState = {
            isBlinking: false,
            isTalking: false,
            currentEmotion: 'neutral',
            animationSpeed: 1,
            blinkInterval: null,
            talkAnimation: null
        };

        // 表情配置
        this.emotions = {
            neutral: {
                mouth: { width: '40px', height: '20px', borderRadius: '0 0 40px 40px' },
                eyes: { transform: 'scaleY(1)' },
                background: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'
            },
            happy: {
                mouth: { width: '50px', height: '25px', borderRadius: '25px 25px 25px 25px' },
                eyes: { transform: 'scaleY(0.8)' },
                background: 'linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%)'
            },
            surprised: {
                mouth: { width: '30px', height: '30px', borderRadius: '50%' },
                eyes: { transform: 'scaleY(1.2)' },
                background: 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)'
            },
            confused: {
                mouth: { width: '35px', height: '15px', borderRadius: '35px 35px 0 0', transform: 'rotate(180deg)' },
                eyes: { transform: 'scaleY(1)' },
                background: 'linear-gradient(135deg, #fd79a8 0%, #e84393 100%)'
            },
            thinking: {
                mouth: { width: '30px', height: '15px', borderRadius: '0 30px 30px 0' },
                eyes: { transform: 'scaleY(0.7)' },
                background: 'linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%)'
            }
        };

        // 初始化
        this.init();
    }

    // 初始化数字人
    init() {
        this.avatarElement = document.getElementById('digitalHuman');
        this.mouthElement = document.getElementById('mouth');
        this.statusElement = document.getElementById('humanStatus');
        this.visualizerElement = document.getElementById('audioVisualizer');
        
        if (this.avatarElement) {
            this.eyeElements = this.avatarElement.querySelectorAll('.eye');
            this.startIdleAnimations();
        }

        console.log('数字人动画系统初始化完成');
    }

    // 开始空闲动画
    startIdleAnimations() {
        // 定期眨眼
        this.animationState.blinkInterval = setInterval(() => {
            if (!this.animationState.isBlinking && !this.animationState.isTalking) {
                this.blink();
            }
        }, 3000 + Math.random() * 2000); // 3-5秒随机眨眼

        // 瞳孔跟随鼠标
        this.setupEyeTracking();
    }

    // 设置眼部跟踪
    setupEyeTracking() {
        if (!this.eyeElements) return;

        document.addEventListener('mousemove', (event) => {
            if (this.animationState.isTalking || this.animationState.isBlinking) return;

            const avatar = this.avatarElement.getBoundingClientRect();
            const centerX = avatar.left + avatar.width / 2;
            const centerY = avatar.top + avatar.height / 2;

            const deltaX = event.clientX - centerX;
            const deltaY = event.clientY - centerY;

            // 限制移动范围
            const maxMove = 3;
            const moveX = Math.max(-maxMove, Math.min(maxMove, deltaX / 50));
            const moveY = Math.max(-maxMove, Math.min(maxMove, deltaY / 50));

            this.eyeElements.forEach(eye => {
                const pupil = eye.querySelector('.pupil');
                if (pupil) {
                    pupil.style.transform = `translate(${moveX}px, ${moveY}px)`;
                }
            });
        });
    }

    // 眨眼动画
    blink() {
        if (this.animationState.isBlinking || !this.eyeElements) return;

        this.animationState.isBlinking = true;
        
        this.eyeElements.forEach(eye => {
            eye.classList.add('blinking');
        });

        setTimeout(() => {
            this.eyeElements.forEach(eye => {
                eye.classList.remove('blinking');
            });
            this.animationState.isBlinking = false;
        }, 300);
    }

    // 设置表情
    setEmotion(emotion) {
        if (!this.emotions[emotion] || !this.avatarElement) return;

        this.animationState.currentEmotion = emotion;
        const config = this.emotions[emotion];

        // 更新头像背景
        const avatar = this.avatarElement.querySelector('.avatar');
        if (avatar) {
            avatar.style.background = config.background;
        }

        // 更新嘴巴
        if (this.mouthElement) {
            Object.assign(this.mouthElement.style, config.mouth);
        }

        // 更新眼睛
        if (this.eyeElements) {
            this.eyeElements.forEach(eye => {
                Object.assign(eye.style, config.eyes);
            });
        }

        console.log(`数字人表情切换为: ${emotion}`);
    }

    // 开始说话动画
    startTalking() {
        if (this.animationState.isTalking || !this.mouthElement) return;

        this.animationState.isTalking = true;
        this.mouthElement.classList.add('talking');
        
        // 显示音频可视化
        if (this.visualizerElement) {
            this.visualizerElement.classList.add('active');
        }

        // 更新状态
        this.updateStatus('正在说话...', 'speaking');

        console.log('开始说话动画');
    }

    // 停止说话动画
    stopTalking() {
        if (!this.animationState.isTalking || !this.mouthElement) return;

        this.animationState.isTalking = false;
        this.mouthElement.classList.remove('talking');
        
        // 隐藏音频可视化
        if (this.visualizerElement) {
            this.visualizerElement.classList.remove('active');
        }

        // 恢复默认表情
        this.setEmotion('neutral');
        this.updateStatus('准备就绪', 'ready');

        console.log('停止说话动画');
    }

    // 开始听取动画
    startListening() {
        // 设置专注表情
        this.setEmotion('thinking');
        this.updateStatus('正在倾听...', 'listening');

        // 显示音频可视化
        if (this.visualizerElement) {
            this.visualizerElement.classList.add('active');
        }

        console.log('开始听取动画');
    }

    // 停止听取动画
    stopListening() {
        // 隐藏音频可视化
        if (this.visualizerElement) {
            this.visualizerElement.classList.remove('active');
        }

        // 恢复默认状态
        this.setEmotion('neutral');
        this.updateStatus('准备就绪', 'ready');

        console.log('停止听取动画');
    }

    // 更新状态显示
    updateStatus(text, type = 'ready') {
        if (!this.statusElement) return;

        const statusText = this.statusElement.querySelector('.status-text');
        const indicator = this.statusElement.querySelector('.pulse-indicator');

        if (statusText) {
            statusText.textContent = text;
        }

        if (indicator) {
            // 移除所有状态类
            indicator.classList.remove('ready', 'listening', 'speaking', 'error');
            indicator.classList.add(type);

            // 根据状态设置颜色
            switch (type) {
                case 'ready':
                    indicator.style.background = '#48bb78';
                    break;
                case 'listening':
                    indicator.style.background = '#4299e1';
                    break;
                case 'speaking':
                    indicator.style.background = '#ed8936';
                    break;
                case 'error':
                    indicator.style.background = '#e53e3e';
                    break;
                default:
                    indicator.style.background = '#718096';
            }
        }
    }

    // 显示思考动画
    showThinking() {
        this.setEmotion('thinking');
        this.updateStatus('正在思考...', 'thinking');

        // 添加思考点动画
        const dots = ['', '.', '..', '...'];
        let index = 0;
        
        const thinkingInterval = setInterval(() => {
            const statusText = this.statusElement?.querySelector('.status-text');
            if (statusText && this.animationState.currentEmotion === 'thinking') {
                statusText.textContent = `正在思考${dots[index % dots.length]}`;
                index++;
            } else {
                clearInterval(thinkingInterval);
            }
        }, 500);
    }

    // 显示错误状态
    showError(message = '出现错误') {
        this.setEmotion('confused');
        this.updateStatus(message, 'error');

        // 3秒后恢复正常状态
        setTimeout(() => {
            this.setEmotion('neutral');
            this.updateStatus('准备就绪', 'ready');
        }, 3000);
    }

    // 播放欢迎动画
    playWelcomeAnimation() {
        const sequence = [
            { emotion: 'happy', duration: 1000 },
            { emotion: 'neutral', duration: 500 }
        ];

        let index = 0;
        const playNext = () => {
            if (index < sequence.length) {
                const step = sequence[index];
                this.setEmotion(step.emotion);
                setTimeout(playNext, step.duration);
                index++;
            }
        };

        this.updateStatus('你好！欢迎使用数字人助手', 'ready');
        playNext();
    }

    // 随机表情动画
    playRandomEmotion() {
        const emotions = Object.keys(this.emotions).filter(e => e !== 'neutral');
        const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)];
        
        this.setEmotion(randomEmotion);
        
        setTimeout(() => {
            this.setEmotion('neutral');
        }, 2000);
    }

    // 同步口型与音频
    syncLipWithAudio(audioData) {
        if (!this.mouthElement || !this.animationState.isTalking) return;

        // 简单的音频数据处理
        if (audioData && audioData.length > 0) {
            const average = audioData.reduce((sum, value) => sum + Math.abs(value), 0) / audioData.length;
            const scale = 1 + (average * 0.5); // 根据音频强度调整嘴巴大小
            
            this.mouthElement.style.transform = `scaleY(${scale})`;
        }
    }

    // 获取当前状态
    getStatus() {
        return {
            emotion: this.animationState.currentEmotion,
            isBlinking: this.animationState.isBlinking,
            isTalking: this.animationState.isTalking,
            animationSpeed: this.animationState.animationSpeed
        };
    }

    // 设置动画速度
    setAnimationSpeed(speed) {
        this.animationState.animationSpeed = Math.max(0.1, Math.min(3, speed));
        
        // 更新CSS动画速度
        if (this.avatarElement) {
            this.avatarElement.style.animationDuration = `${4 / this.animationState.animationSpeed}s`;
        }
    }

    // 重置到默认状态
    reset() {
        this.stopTalking();
        this.stopListening();
        this.setEmotion('neutral');
        this.updateStatus('准备就绪', 'ready');
    }

    // 销毁动画系统
    destroy() {
        // 清理定时器
        if (this.animationState.blinkInterval) {
            clearInterval(this.animationState.blinkInterval);
        }

        // 清理动画
        this.reset();

        // 移除事件监听器
        document.removeEventListener('mousemove', this.setupEyeTracking);

        console.log('数字人动画系统已销毁');
    }
}

// 导出数字人实例
const digitalHuman = new DigitalHuman();

// 如果在Node.js环境中，导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DigitalHuman;
}