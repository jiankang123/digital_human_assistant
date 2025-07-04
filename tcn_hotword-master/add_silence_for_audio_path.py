from pydub import AudioSegment
import os
import sys

# 设置静音时长
silence_duration = 500 # 单位为毫秒

# 设置音频目录路径
audio_dir = sys.argv[1]
outaudio_dir = sys.argv[2]
hotword = sys.argv[3]
os.mkdir(outaudio_dir)
fout = open(os.path.join(outaudio_dir, "trans.txt"), 'w', encoding='utf-8')

# 遍历目录下的所有音频文件
for filename in os.listdir(audio_dir):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        # 加载音频文件
        audio_path = os.path.join(audio_dir, filename)
        audio = AudioSegment.from_file(audio_path)

        # 创建静音段
        silence = AudioSegment.silent(duration=silence_duration)

        # 在音频开头和结尾添加静音段
        audio_with_silence = silence + audio + silence

        # 保存修改后的音频文件
        output_path = os.path.join(outaudio_dir, filename)
        audio_with_silence.export(output_path, format="wav")
        
        fout.write(output_path + '\t' + hotword + '\n')
