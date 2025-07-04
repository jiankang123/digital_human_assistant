import os, sys
import argparse
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import math
import multiprocessing as mlp
from tqdm import tqdm
'''
1.volume 
2.speed
3.pitch
4.reverb
5.rir
6.noise
'''

# @profile
def perturb_utt(utt, rd, args, rir_utts, noise_utts, repeat_i):
    raw_audio, label = utt
    gain_prob, gain_min, gain_max = args.gain_prob, args.gain_min, args.gain_max
    speed_prob, speed_min, speed_max = args.speed_prob, args.speed_min, args.speed_max
    pitch_prob, pitch_min, pitch_max = args.pitch_prob, args.pitch_min, args.pitch_max
    reverb_prob, reverb_min, reverb_max = args.reverb_prob, args.reverb_min, args.reverb_max
    rir_prob, noise_prob, snr_min, snr_max = args.rir_prob, args.noise_prob, args.snr_min, args.snr_max
    compress_prob, compress_min, compress_max = args.compress_prob, args.compress_min, args.compress_max

    effects = []
    wav_name = os.path.basename(raw_audio).replace('.wav', '')
    if gain_prob > 0 and rd.random() <= gain_prob:
        gain = rd.randint(gain_min, gain_max)
        effects.append(['gain', f'{gain}'])
        wav_name += f'_gain_{gain}'
    if speed_prob > 0 and rd.random() <= speed_prob:
        speed = rd.randint(speed_min*100, speed_max*100) / 100
        effects.append(["speed", f"{speed}"])
        wav_name += f'_speed_{speed}'
    if pitch_prob > 0 and rd.random() <= pitch_prob:
        pitch = rd.randint(pitch_min, pitch_max)
        effects.append(['pitch', f"{pitch}"])
        wav_name += f'_pitch_{pitch}'
    if reverb_prob > 0 and rd.random() <= reverb_prob:
        reverb = rd.randint(reverb_min, reverb_max)
        effects.append(["reverb", "-w", f"{reverb}"])
        wav_name += f'_reverb_{reverb}'

    effects.append(["rate", "16000"])
    if len(effects) == 0:
        waveform, sr = torchaudio.load(raw_audio)
    else:
        waveform, sr = torchaudio.sox_effects.apply_effects_file(
            raw_audio, effects)
    assert sr == 16000

    if rir_prob > 0 and rd.random() <= rir_prob:
        rir_utt = rd.choice(rir_utts)
        rir_raw, rir_sr = torchaudio.load(rir_utt)
        rir_16k = F.resample(rir_raw, rir_sr, 16000)
        rir = rir_16k[:, 0 : int(rir_sr * 0.3)]
        rir = rir / torch.norm(rir, p=2)
        RIR = torch.flip(rir, [1])
        waveform_ = torch.nn.functional.pad(waveform, (RIR.shape[1] - 1, 0))
        # cost many time 
        waveform = torch.nn.functional.conv1d(waveform_[None, ...], RIR[None, ...])[0]
        wav_name += f'_rir'

    if noise_prob > 0 and rd.random() <= noise_prob:
        # cost many time 
        noise_utt = rd.choice(noise_utts)
        snr_db = rd.randint(snr_min, snr_max)
        noise, noise_sr = torchaudio.load(noise_utt)
        assert noise_sr == 16000
        if noise.shape[1] >= waveform.shape[1]:
            noise = noise[:, : waveform.shape[1]]
        else:
            time = math.ceil(waveform.shape[1] / noise.shape[1])
            noise = noise.repeat([1, time])[:, : waveform.shape[1]]
        speech_rms = waveform.norm(p=2)
        noise_rms = noise.norm(p=2)
        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / speech_rms
        # waveform = (scale * waveform + noise) / 2
        waveform = (1/scale * noise + waveform) / 2
        wav_name += f'_snr_{snr_db}'

    if compress_prob > 0 and rd.random() <= compress_prob:
        # compress_level = rd.randint(compress_min, compress_max)
        # config = {"format": "gsm"}
        # waveform = F.apply_codec(waveform, sr, **config)
        # config = {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8}
        # waveform = F.apply_codec(waveform, sr, **config)
        # config = {"format": "vorbis", "compression": compress_level}
        # waveform = F.apply_codec(waveform, sr, **config)
        # config = {"format": "wav", "encoding": "PCM_S"}
        # waveform = F.apply_codec(waveform, sr, **config)
        # wav_name += f'_cmp_{compress_level}'
        pass


    wav_path = os.path.join(args.output_dir, wav_name+f'_re{repeat_i}.wav')
    torchaudio.backend.sox_io_backend.save(wav_path, waveform, 16000,
            format='wav', 
            encoding="PCM_S", 
            bits_per_sample=16)
    return (wav_path, label)

# @profile
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TODO support config file
    parser.add_argument('--config',
                        type=str,
                        default=None,
                        help='config yaml file')
    parser.add_argument('--input_transcript',
                        type=str,
                        help='Input transcript file.',
                        default='/ssd1',
                        )
    parser.add_argument('--output_dir',
                        type=str,
                        help='Directory to save generated wav file.',
                        default='/ssd1',
                        )
    parser.add_argument('--output_transcript',
                        type=str,
                        help='Output transcript file.',
                        default='/ssd1',
                        )
    # 增益音频
    parser.add_argument('--gain_prob',
                        type=float,
                        default=0,
                        help='prob of gain audio')
    parser.add_argument('--gain_min',
                        type=int,
                        default=-20,
                        help='Gain min(db).')
    parser.add_argument('--gain_max',
                        type=int,
                        default=0,
                        help='Gain max(db).')
    # 变速
    parser.add_argument('--speed_prob',
                        type=float,
                        default=0,
                        help='prob of speed audio')
    parser.add_argument('--speed_min',
                        type=float,
                        help='Speed min.',
                        default=0.9)
    parser.add_argument('--speed_max',
                        type=float,
                        help='Speed max.',
                        default=1.1)
    # 变调
    parser.add_argument('--pitch_prob',
                        type=float,
                        default=0,
                        help='prob of shift audio pitch')
    parser.add_argument('--pitch_min',
                        type=int,
                        help='pitch shift min.',
                        default=-5)
    parser.add_argument('--pitch_max',
                        type=int,
                        help='pitch shift max.',
                        default=5)
    # 加混响（算法模拟混响）
    parser.add_argument('--reverb_prob',
                        type=float,
                        default=0,
                        help='prob of reverb audio')
    parser.add_argument('--reverb_min',
                        type=int,
                        help='reverb strength.',
                        default=20)
    parser.add_argument('--reverb_max',
                        type=int,
                        help='reverb strength.',
                        default=40)        
    # rir
    parser.add_argument('--rir_prob',
                        type=float,
                        default=0,
                        help='prob of apply rir on audio')
    parser.add_argument('--rir_transcript',
                        type=str,
                        help='rir wav transcript file.')

    # 加噪声，信噪比
    parser.add_argument('--noise_prob',
                        type=float,
                        default=0,
                        help='prob of add noise')
    parser.add_argument('--snr_min',
                        type=int,
                        default=0,
                        help='SNR config')
    parser.add_argument('--snr_max',
                        type=int,
                        default=5,
                        help='SNR config')
    parser.add_argument('--noise_transcript',
                        type=str,
                        help='noise wav transcript file.')
    
    # 压缩处理，压缩等级
    parser.add_argument('--compress_prob',
                        type=float,
                        default=0,
                        help='prob of add noise')
    parser.add_argument('--compress_min',
                        type=int,
                        default=-1,
                        help='min compress level')
    parser.add_argument('--compress_max',
                        type=int,
                        default=10,
                        help='max compress level')
    
    parser.add_argument('--repeat', type=int, default=1, help='repeat times')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of process workers.')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    raw_utts = []
    with open(args.input_transcript, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            if '\t' in line:
                wav_path, other = line.strip().split('\t', 1)
            else:
                wav_path, other = line.strip().split(' ', 1)
            raw_utts.append((wav_path, other))

    rir_utts = []
    if args.rir_transcript:
        rir_path_list = []
        if ',' in args.rir_transcript:
            rir_path_list = args.rir_transcript.split(',')
        else:
            rir_path_list = [args.rir_transcript]
        for each_rir_path in rir_path_list:
            with open(each_rir_path, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    if '\t' in line:
                        wav_path = line.strip().split('\t', 1)[0]
                    elif ' ' in line:
                        wav_path = line.strip().split(' ', 1)[0]
                    else:
                        wav_path = line.strip()
                    rir_utts.append(wav_path)

    noise_utts = []
    if args.noise_transcript:
        noise_path_list = []
        if ',' in args.noise_transcript:
            noise_path_list = args.noise_transcript.split(',')
        else:
            noise_path_list = [args.noise_transcript]
        for each_noise_path in noise_path_list:
            with open(each_noise_path, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    if '\t' in line:
                        wav_path = line.strip().split('\t', 1)[0]
                    elif ' ' in line:
                        wav_path = line.strip().split(' ', 1)[0]
                    else:
                        wav_path = line.strip()
                    noise_utts.append(wav_path)

    all_perturbed_utts = []
    
    print("Preparing tasks ...")
    tasks = []
    pool = mlp.Pool(args.num_workers)
    for repeat_i in range(args.repeat):
        for utt in raw_utts:
            rd = np.random.RandomState()
            tasks.append(
                pool.apply_async(perturb_utt, (utt, rd, args, rir_utts, noise_utts, repeat_i))
            )

    print("Doing tasks ...")
    for i in tqdm(range(len(tasks))):
        all_perturbed_utts.append(tasks[i].get())



    utts = [u for u in all_perturbed_utts if u is not None]
    output_transcript = os.path.join(args.output_dir, 'trans.txt')
    with open(output_transcript, 'w', encoding='utf-8') as f:
        for wav, other in utts:
            f.write('{}\t{}\n'.format(wav, other))

    print(f"Done! Transcript is in {output_transcript}")


if __name__ == "__main__":
    main()