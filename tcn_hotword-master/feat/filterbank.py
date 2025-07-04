import torch
import numpy as np
from feat_param import povey_win_fn, mel_filter, mel_filter_weight

LB_START_BANDS = 0
LB_END_BANDS = 256
LB_FBANK_NUM = 23


def compute_fbank(signal, coeff=0.97, fft_size=512):
    
    # dc remove
    padding = np.zeros(240)
    signal = np.append(padding,signal[:160])

    signal_len = len(signal)
    signal -= signal.mean()
    
    # pre-emphasis
    emphasis_list = np.insert(signal[1:] - coeff * signal[:-1], 0, signal[0] - signal[0] * coeff)
    signal = torch.tensor(emphasis_list)
    
    # window
    frames = []
    if signal_len < 400:
        frame_num = 0
    else:
        frame_num = ((signal_len-400) // 160) + 1
    for i in range(frame_num):
        temp = signal[i*160:i*160+400]
        frames.append(temp * povey_win_fn)

    # fft
    signal_power_list = []
    for frame in frames:
        signal_freq = torch.fft.fft(frame, 512).numpy()[:256]
        freq_power = np.zeros(256)
        for i in range(256):
            freq_power[i] = np.real(signal_freq[i]) * np.real(signal_freq[i]) + np.imag(signal_freq[i]) * np.imag(signal_freq[i])
        signal_power_list.append(torch.tensor(freq_power))

    # mel-filtering
    fbank_buf_list = []
    for freq_power in signal_power_list:
        fbank_buf = torch.zeros(LB_FBANK_NUM)
        weight_idx = 0
        for i in range(LB_FBANK_NUM):
            bin_begin_idx = mel_filter[i][0]
            bin_len = mel_filter[i][1]
            temp_sum = 0
            for j in range(bin_len):
                temp_sum += freq_power[bin_begin_idx] * mel_filter_weight[weight_idx]
                bin_begin_idx += 1
                weight_idx += 1
            fbank_buf[i] = temp_sum
        fbank_buf_list.append(fbank_buf)

    # log
    fbank_list = []
    for fbank_buf in fbank_buf_list:
        fbank = torch.zeros_like(fbank_buf)
        for i in range(LB_FBANK_NUM):
            fbank[i] = fbank_buf[i] if fbank_buf[i] == 0 else torch.log(fbank_buf[i])
        fbank_list.append(fbank)
    print(fbank)
    return fbank
