import os
import pywebio
import pywebio.output as output
import pywebio.input as input
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pydub import AudioSegment
from feat import compute_fbank, get_CMVN

def check_wav_format(wav_path):
    audio = AudioSegment.from_wav(wav_path)
    if not audio.sample_width == 2:
        return False
    if not audio.frame_rate == 16000:
        return False
    if not audio.channels == 1:
        return False
    return True

def wav_test(wav_path, net, mean, scale, output_dir=None, plot=True):
    fbank = compute_fbank(wav_path)
    fbank = (fbank - mean) * scale
    pred = net(fbank.unsqueeze(0)).squeeze(0).detach().numpy()

    if plot:
        pic_path = plot_result(wav_path, pred, output_dir)
    return pic_path


def plot_result(wav_path, pred_result, output_dir=None):
    garbage_confidence = pred_result[:,0]
    num_classes = pred_result.shape[-1]
    pred_seq = pred_result.argmax(-1).tolist()
    # print(pred_seq)
    fig = plt.figure()
    if wav_path.lower().endswith('.wav'):
        wav_data = np.memmap(wav_path, dtype='h', offset=44, mode='r')
    else:
        wav_data = np.memmap(wav_path, dtype='h', offset=0, mode='r')

    det_time = np.linspace(0, len(wav_data), len(pred_seq))
    ax = fig.add_subplot(111)
    ax.plot(wav_data)
    ax2 = ax.twinx()
    ax2.plot(det_time, pred_seq, '-r')
    ax2.plot(det_time, garbage_confidence, '-y')
    ax2.grid()
    ax.set_xlabel("time")
    ax.set_ylabel("sample")
    ax2.set_ylabel("class")
    ax2.axis([0, len(wav_data), 0, num_classes])
    if not output_dir:
        pic_path = 'fig_dir/fig_{}.png'.format(os.path.basename(wav_path))
    else:
        pic_path = os.path.join(output_dir, 'fig_{}.png'.format(os.path.basename(wav_path)))
    fig.savefig(pic_path)
    return pic_path


def main():
    wav_path = 'server/tmp.wav'
    output.put_markdown('# 唤醒&快词模型演示')
    output.put_markdown('使用说明：')
    output.put_markdown("""
                        - step1: 选择模型
                        - step2: 选择音频（16k、16bit、mono）
                        - step3: 显示结果,红线表示识别结果，对应类别在右y轴
                        """)
    # step 1
    model_list = [
        "婴儿哭声检测",
        "insta中文快词",
        "insta英文快词",
        "vivo中文快词",
        "vivo英文快词"
    ]
    model = input.select(label="step1: 选择模型", options=model_list)
    if model == "insta中文快词":
        net_path = "data/insta_cn/exps/zk_20220607/exp_big_avgpool/epoch38_0.9464.pt"
        cmvn = "data/insta_cn/exps/zk_20220607/data/cmvn.json"
        mean, scale = get_CMVN(cmvn)
        net = torch.load(net_path, map_location='cpu').eval()
    elif model == "insta英文快词":
        net_path = "data/insta_cn/exps/zk_20220607/exp_big_avgpool/epoch38_0.9464.pt"
        cmvn = "data/insta_cn/exps/zk_20220607/data/cmvn.json"
        mean, scale = get_CMVN(cmvn)
        net = torch.load(net_path, map_location='cpu').eval()
    elif model == "婴儿哭声检测":
        net_path = "data/baby_cry/exps/zk_20220615/exp_small_avgpool/epoch8_0.9915.pt"
        cmvn = "data/baby_cry/exps/zk_20220615/data/cmvn.json"
        mean, scale = get_CMVN(cmvn)
        net = torch.load(net_path, map_location='cpu').eval()
    elif model == "vivo中文快词":
        net_path = "data/insta_cn/exps/zk_20220607/exp_big_avgpool/epoch38_0.9464.pt"
        cmvn = "data/insta_cn/exps/zk_20220607/data/cmvn.json"
        mean, scale = get_CMVN(cmvn)
        net = torch.load(net_path, map_location='cpu').eval()
    elif model == "vivo英文快词":
        net_path = "data/insta_cn/exps/zk_20220607/exp_big_avgpool/epoch38_0.9464.pt"
        cmvn = "data/insta_cn/exps/zk_20220607/data/cmvn.json"
        mean, scale = get_CMVN(cmvn)
        net = torch.load(net_path, map_location='cpu').eval()


    # step 2
    file = input.file_upload('step2: 选择音频', '.wav')
    with open(wav_path, 'wb') as fin:
        fin.write(file['content'])
    if not check_wav_format(wav_path):
        output.put_markdown('Error audio format!!!')
    
    # step 3
    pic_path = wav_test(wav_path, net, mean, scale)
    # print(pic_path)
    output.put_image(open(pic_path, 'rb').read(), format='png', title='step 3: 显示结果')



if __name__=='__main__':
    pywebio.start_server(main, host="0.0.0.0", port=11111)