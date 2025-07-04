import os
import sys
from wav_process import pcm2wav
from split_insta_wav import *
from kws_split import apply_onex3_kws


if __name__ == '__main__':
    pcm_input_path = sys.argv[1]
    pcm_4c_path = sys.argv[2]
    pcm_2c_path = sys.argv[3]
    # 需要第三个通道增益就设置成0，不需要第三个通道增益就设置成1
    mic_type = sys.argv[4]

    for root,dirs,files in os.walk(pcm_input_path):
        for f in files:
            if f.endswith('.PCM'):
                pcm_path = os.path.join(root,f)
                print(pcm_path)
                voice_data = AudioSegment.from_file(
                        file=pcm_path, sample_width=4, frame_rate=48000,channels=4,) 
                if not mic_type:
                    gain_voice_data = gain_channel2(voice_data)
                else:
                    gain_voice_data = voice_data
                wav_name = f[:-3]+'wav'
                out_4c_path = os.path.join(pcm_4c_path,wav_name)
                gain_voice_data.export(out_4c_path,format='wav')
                kws_wav_path = os.path.join(pcm_2c_path, wav_name.replace('.wav', '_kws_out.wav'))
                # 需要增益的输出2通道mic_kws结果
                apply_onex3_kws(out_4c_path,kws_wav_path,mic_type)
                # 不需要增益就划分单通道
                if mic_type:
                    audio = AudioSegment.from_wav(kws_wav_path)
                    audios = audio.split_to_mono()
                    for i, au in enumerate(audios):
                        au.export(kws_wav_path.replace('_2','_1').replace(".wav", "_{}.wav".format(i)), format="wav")



    # pcm2wav(pcm_path)