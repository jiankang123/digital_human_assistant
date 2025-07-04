import torch
import numpy as np
import threading
from torch_mel_scale_win import povey_win_fn, mel_scale

LB_START_BANDS = 0
LB_END_BANDS = 256
LB_FBANK_NUM = 23



def frames_fbank(frames, fbank_dic, thread_index, frames_per_thread, fft_size, coeff, mel_scale):
    fbank_list = []
    for frame_index in range(frames_per_thread):
        # 分帧
        frame = frames[frame_index*160:frame_index*160+400].clone()

        # dc remove, 减均值
        frame -= frame.mean()

        # pre-emphasis, 预加重
        frame[0] = frame[0] - frame[0] * coeff
        frame[1:] = frame[1:] - coeff * frame[:-1]
        if thread_index == 0 and frame_index == 1:
            print('pre-emphasis',frame)
        # window, 加窗
        win_frame = frame * povey_win_fn

        # fft, 快速离散傅里叶变换
        fft_256_frame = torch.fft.fft(win_frame,fft_size)[:256]
        freq_power = fft_256_frame.abs().pow(2.0)
        
        # mel-filtering, mel滤波
        fbank_buf = torch.sum(mel_scale * freq_power,dim=-1)

        # log, 取log
        fbank = fbank_buf + 1e-6
        fbank_list.append(fbank.log())
    
    fbank_dic[thread_index] = torch.stack(fbank_list,dim=0)



'''
对一条语音，分6个线程，每个线程线性处理帧
'''
def compute_fbank(signal, coeff=0.97, fft_size=512):
    
    # padding wavform
    padding_head = torch.zeros(240,dtype=torch.float)
    padding_rear = torch.zeros(160,dtype=torch.float)
    signal = torch.cat([padding_head,signal],dim=0)
    signal_len = signal.shape[-1]
    # print(signal[160:560])

    # split frame, 分帧
    if signal_len < 400:
        frame_num = 0
    else:
        frame_num = ((signal_len-400)//160)+1

    # thread
    thread_num=6
    fbank_dic = {}
    th_list = []
    frames_per_thread = frame_num//thread_num

    if frames_per_thread == 0:
        thread_num = frame_num
        frames_per_thread = 1
    for thread_index in range(thread_num):
        start = thread_index * frames_per_thread * 160
        if thread_index == (thread_num - 1):
            end = (frame_num-1) * 160 + 400
        else:
            end = (((thread_index+1) * frames_per_thread)-1) * 160 + 400
        th = threading.Thread(target=frames_fbank, args=(signal[start:end],fbank_dic,thread_index,frames_per_thread,fft_size,coeff, mel_scale))
        th.start()
        th_list.append(th)
    for th in th_list:
        th.join()

    ret_fbank = []

    for i in range(thread_num):
        ret_fbank.append(fbank_dic[i])
    if (signal_len-400) % 160 != 0:
        rest_signal = signal[frame_num*160:frame_num*160+400]
        frames_fbank(rest_signal, fbank_dic, -1, 1)
        ret_fbank.append(fbank_dic[-1])

    ret_fbank = torch.cat(ret_fbank,dim=0)

    return ret_fbank
