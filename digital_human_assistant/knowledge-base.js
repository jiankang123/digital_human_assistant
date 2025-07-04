// 知识库管理器
class KnowledgeBase {
    constructor() {
        this.qaDatabase = [
            // 基本问候
            {
                keywords: ['你好', 'hello', '您好', '嗨', 'hi'],
                responses: [
                    '你好！我是您的数字人助手，很高兴为您服务！',
                    '您好！有什么可以帮助您的吗？',
                    '嗨！欢迎来到数字人交互系统！'
                ],
                category: 'greeting'
            },
            {
                keywords: ['再见', 'goodbye', '拜拜', '谢谢', 'bye'],
                responses: [
                    '再见！期待下次为您服务！',
                    '谢谢您的使用，祝您愉快！',
                    '拜拜！有需要随时找我！'
                ],
                category: 'farewell'
            },

            // 功能介绍
            {
                keywords: ['你能做什么', '功能', '能力', '可以做', '会什么'],
                responses: [
                    '我可以与您进行语音对话，回答各种问题，还能帮您查询信息、聊天交流等。我支持语音识别和语音合成，让我们的交流更自然！',
                    '我的主要功能包括：语音交互、智能问答、信息查询、日常聊天等。您可以用语音或文字和我交流！',
                    '我是一个多功能的数字助手，可以帮您解答问题、进行对话、提供信息服务等。试试和我聊聊吧！'
                ],
                category: 'capability'
            },

            // 天气相关
            {
                keywords: ['天气', '气温', '下雨', '晴天', '阴天', '温度'],
                responses: [
                    '很抱歉，我目前还无法实时获取天气信息。建议您查看天气预报App或询问语音助手获取准确的天气信息。',
                    '我暂时不能提供实时天气数据，您可以通过手机天气应用或搜索引擎查询当地天气情况。',
                    '天气查询功能正在开发中，目前请通过其他渠道获取天气信息。感谢理解！'
                ],
                category: 'weather'
            },

            // 时间相关
            {
                keywords: ['时间', '几点', '现在', '日期', '今天'],
                responses: [
                    `现在是 ${new Date().toLocaleString('zh-CN')}`,
                    `当前时间：${new Date().toLocaleTimeString('zh-CN')}，今天是 ${new Date().toLocaleDateString('zh-CN')}`,
                    `今天是 ${new Date().toLocaleDateString('zh-CN')}，现在时间是 ${new Date().toLocaleTimeString('zh-CN')}`
                ],
                category: 'time'
            },

            // 技术相关
            {
                keywords: ['技术', '开发', '编程', '代码', 'AI', '人工智能'],
                responses: [
                    '我基于Web技术开发，使用了语音识别API、语音合成API等现代浏览器技术，结合智能问答系统为您提供服务。',
                    '这个数字人系统采用了HTML5、CSS3、JavaScript等前端技术，集成了语音交互功能，让人机对话更加自然。',
                    '我使用了现代Web标准和AI技术，包括语音识别、自然语言处理等，致力于提供更好的用户体验。'
                ],
                category: 'technology'
            },

            // 帮助相关
            {
                keywords: ['帮助', 'help', '怎么用', '如何使用', '教程'],
                responses: [
                    '使用很简单！您可以点击麦克风按钮说话，或者直接在文本框输入问题。我会用语音和文字回复您。还可以点击快捷问题按钮快速开始对话！',
                    '操作指南：1. 点击语音按钮开始说话 2. 或在文本框输入问题 3. 可以使用快捷问题按钮 4. 点击设置按钮调整语音参数',
                    '您可以通过多种方式与我交流：语音输入、文字输入或点击预设问题。右下角的设置按钮可以调整语音效果哦！'
                ],
                category: 'help'
            },

            // 情感相关
            {
                keywords: ['心情', '开心', '难过', '高兴', '烦恼', '压力'],
                responses: [
                    '我理解您的感受。如果您需要倾诉或聊天，我很乐意陪伴您。有什么想分享的吗？',
                    '每个人都会有各种情绪，这很正常。我在这里倾听，您可以和我分享任何想法。',
                    '情绪的起伏是生活的一部分。如果您愿意，可以告诉我更多，我会认真倾听的。'
                ],
                category: 'emotion'
            },

            // 学习相关
            {
                keywords: ['学习', '知识', '教育', '课程', '学校'],
                responses: [
                    '学习是终身的过程！我可以陪您讨论各种话题，虽然我的知识有限，但我们可以一起探索和思考。',
                    '学习很重要！虽然我无法替代专业教育，但我愿意在日常交流中与您分享想法和观点。',
                    '持续学习很棒！我们可以通过对话交流想法，相互启发。有什么特别想了解的话题吗？'
                ],
                category: 'learning'
            },

            // 默认回复
            {
                keywords: ['default'],
                responses: [
                    '我理解您的意思，但可能无法给出完美的回答。您可以换个方式问我，或者问一些其他问题。',
                    '这是一个有趣的问题！虽然我的知识有限，但我们可以继续聊聊其他话题。',
                    '抱歉我可能没有完全理解您的问题。您可以尝试用不同的方式表达，或者问问我能做什么。',
                    '我会继续学习和改进的！现在我们聊点其他的吧，比如您今天过得怎么样？'
                ],
                category: 'default'
            }
        ];

        // 预加载随机回复索引
        this.responseIndices = {};
        this.qaDatabase.forEach((qa, index) => {
            this.responseIndices[index] = 0;
        });
    }

    // 智能匹配用户输入
    findAnswer(userInput) {
        if (!userInput || userInput.trim() === '') {
            return this.getRandomResponse('default');
        }

        const input = userInput.toLowerCase().trim();
        let bestMatch = null;
        let bestScore = 0;

        // 遍历知识库寻找最佳匹配
        for (let i = 0; i < this.qaDatabase.length; i++) {
            const qa = this.qaDatabase[i];
            if (qa.category === 'default') continue;

            let score = 0;
            const keywords = qa.keywords;

            // 精确匹配得分更高
            for (let keyword of keywords) {
                if (input.includes(keyword.toLowerCase())) {
                    score += keyword.length; // 更长的关键词权重更高
                }
            }

            // 模糊匹配
            if (score === 0) {
                for (let keyword of keywords) {
                    if (this.fuzzyMatch(input, keyword.toLowerCase())) {
                        score += keyword.length * 0.5; // 模糊匹配权重较低
                    }
                }
            }

            if (score > bestScore) {
                bestScore = score;
                bestMatch = qa;
            }
        }

        // 如果找到匹配，返回回复；否则返回默认回复
        if (bestMatch && bestScore > 0) {
            return this.getRandomResponse(bestMatch, this.qaDatabase.indexOf(bestMatch));
        } else {
            return this.getRandomResponse('default');
        }
    }

    // 模糊匹配算法
    fuzzyMatch(text, keyword) {
        // 简单的包含检查和编辑距离
        if (text.includes(keyword)) return true;
        
        // 计算编辑距离
        const distance = this.levenshteinDistance(text, keyword);
        const maxLength = Math.max(text.length, keyword.length);
        const similarity = 1 - distance / maxLength;
        
        return similarity > 0.6; // 相似度阈值
    }

    // 计算编辑距离
    levenshteinDistance(str1, str2) {
        const matrix = [];
        
        for (let i = 0; i <= str2.length; i++) {
            matrix[i] = [i];
        }
        
        for (let j = 0; j <= str1.length; j++) {
            matrix[0][j] = j;
        }
        
        for (let i = 1; i <= str2.length; i++) {
            for (let j = 1; j <= str1.length; j++) {
                if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1,
                        matrix[i][j - 1] + 1,
                        matrix[i - 1][j] + 1
                    );
                }
            }
        }
        
        return matrix[str2.length][str1.length];
    }

    // 获取随机回复
    getRandomResponse(category, qaIndex = null) {
        let responses;
        let index;

        if (typeof category === 'string') {
            // 根据类别查找
            const qa = this.qaDatabase.find(item => item.category === category);
            if (!qa) return '抱歉，我暂时无法回答这个问题。';
            responses = qa.responses;
            index = this.qaDatabase.indexOf(qa);
        } else {
            // 直接传入QA对象
            responses = category.responses;
            index = qaIndex;
        }

        // 轮换回复，避免重复
        const responseIndex = this.responseIndices[index] % responses.length;
        this.responseIndices[index] = (this.responseIndices[index] + 1) % responses.length;

        return responses[responseIndex];
    }

    // 添加新的问答对
    addQA(keywords, responses, category = 'custom') {
        this.qaDatabase.push({
            keywords: Array.isArray(keywords) ? keywords : [keywords],
            responses: Array.isArray(responses) ? responses : [responses],
            category: category
        });

        // 更新响应索引
        const newIndex = this.qaDatabase.length - 1;
        this.responseIndices[newIndex] = 0;
    }

    // 获取所有类别
    getCategories() {
        return [...new Set(this.qaDatabase.map(qa => qa.category))];
    }

    // 根据类别获取问答对
    getQAByCategory(category) {
        return this.qaDatabase.filter(qa => qa.category === category);
    }

    // 搜索问答对
    searchQA(searchTerm) {
        const term = searchTerm.toLowerCase();
        return this.qaDatabase.filter(qa => {
            return qa.keywords.some(keyword => 
                keyword.toLowerCase().includes(term)
            ) || qa.responses.some(response => 
                response.toLowerCase().includes(term)
            );
        });
    }

    // 获取知识库统计信息
    getStats() {
        return {
            totalQAs: this.qaDatabase.length,
            categories: this.getCategories().length,
            totalKeywords: this.qaDatabase.reduce((sum, qa) => sum + qa.keywords.length, 0),
            totalResponses: this.qaDatabase.reduce((sum, qa) => sum + qa.responses.length, 0)
        };
    }
}

// 导出知识库实例
const knowledgeBase = new KnowledgeBase();

// 如果在Node.js环境中，导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KnowledgeBase;
}