#!/usr/bin/env python3
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: yongzou@mobvoi.com (Yong Zou), shenli@mobvoi.com (Shen Li)



import argparse
from concurrent.futures import ProcessPoolExecutor
from distutils.command import clean
import functools
import os
from scipy.io import wavfile
import random
import yaml
import sox
import numpy as np
import math


def _perturb_utt(utt, args, noise_utts):
    raw_audio, label = utt
    raw_audio_filename = os.path.splitext(os.path.basename(raw_audio))[0]
    tns = sox.Transformer()
    # sox.Transformer() official website: https://pysox.readthedocs.io/en/latest/api.html
    surfix = ''
    if args.normalize:
        gain_db = random.randint(args.gain_min, args.gain_max)
        # params_list
        # - gain_db
        # - normalize
        # - limiter
        # - balance
        tns.gain(gain_db=gain_db)
        surfix += '_gain_' + str(gain_db)
    if args.tempo:
        speed = round(random.uniform(args.speed_min, args.speed_max), 3)
        # params_list
        # - factor
        # - audio_type
        # - quick
        tns.tempo(speed)
        surfix += '_tempo_' + str(speed)
    if args.pitch:
        shift = round(random.uniform(args.shift_min, args.shift_max), 3)
        # params_list
        # - n_semitones
        # - quick
        tns.pitch(shift)
        surfix += '_pitch_' + str(shift)
    if args.reverb:
        # 混响强度，默认是40，但加出来的效果和背景噪声有明显的不同，音频信号的混响特别明显，噪声却没有混响
        # 强度为20的时候两个合起来效果还可以
        reverbs_strength = random.randint(args.reverberance_min, args.reverberance_max)
        # params_list
        # - reverberance
        # - high_freq_damping
        # - room_scale
        # - stereo_depth
        # - pre_delay
        # - wet_gain
        # - wet_only  
        tns.reverb(reverberance=reverbs_strength)
        surfix += '_reverb_{}'.format(reverbs_strength)
    perturbed_audio = os.path.join(args.output_dir,
                                   raw_audio_filename + surfix + '.wav')
    if os.path.isfile(perturbed_audio):
        return None
    tns.build(raw_audio, perturbed_audio)

    # 干净平均幅值，用于计算SNR
    SNR = random.randint(args.snr_min, args.snr_max)
    factor = 10 ** (SNR / 20)
    sr, clean_data = wavfile.read(perturbed_audio)
    # 干净音频的平均幅值
    # clean_amp = np.sum(np.absolute(clean_data),axis=0)
    # L2计算
    clean_amp = np.linalg.norm(clean_data)
    if clean_amp == 0:
        print(f"*********{raw_audio}*********")

    if noise_utts:
        tns_noise = sox.Transformer()
        ori_duration = sox.file_info.duration(perturbed_audio)
        noise_audio, _ = random.choice(noise_utts)
        noise_duration = sox.file_info.duration(noise_audio)
        if noise_duration > ori_duration:
            start_location = round(
                random.uniform(0, noise_duration - ori_duration), 3)
            end_location = start_location + ori_duration
        else:
            print('Noise audio {} is too shrot for perturbed audio {}.'.format(
                noise_audio, perturbed_audio))
            start_location = 0
            end_location = noise_duration
        tns_noise.trim(start_location, end_location)

        # 噪声平均幅值，用于计算SNR
        sr, noise_data = wavfile.read(noise_audio)
        temp = noise_data[int(start_location*sr):int((end_location)*sr)]
        # 噪声音频截取部分的平均幅值
        # noise_amp = np.sum(np.absolute(temp), axis=0)
        noise_amp = np.linalg.norm(temp)
        # 噪声音频应该增益的百分比
        gain_persent = clean_amp / noise_amp*factor
        # 噪声音频增益百分比转为增益db
        # 转换关系和ocenaudio的‘效果器->增益’里百分比和db的转换关系相同
        gain_db = math.floor(20 * math.log10(gain_persent))

        tns_noise.gain(gain_db=gain_db)
        noise_trim_audio = os.path.join(
            args.output_dir, raw_audio_filename + ' _noise_trim.wav')
        tns_noise.build(noise_audio, noise_trim_audio)
        # 后缀里snr后跟的数字就是信噪比
        surfix += '_snr_{}_'.format(SNR) + os.path.splitext(
            os.path.basename(noise_audio))[0] + '_' + str(start_location)
        noise_mixed_audio = os.path.join(args.output_dir,
                                         raw_audio_filename + surfix + '.wav')
        cbn = sox.Combiner()
        cbn.build([perturbed_audio, noise_trim_audio], noise_mixed_audio,
                  'mix')
        os.remove(noise_trim_audio)
        os.remove(perturbed_audio)
        perturbed_audio = noise_mixed_audio
    return (perturbed_audio, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    # 信噪比
    parser.add_argument('--snr_min',
                        type=int,
                        default=0,
                        help='SNR config')
    parser.add_argument('--snr_max',
                        type=int,
                        default=5,
                        help='SNR config')
    # 增益音频
    parser.add_argument('--normalize',
                        dest='normalize',
                        action='store_true',
                        help='gain wav.(default enabled)')
    parser.add_argument('--no-normalize',
                        dest='normalize',
                        action='store_false',
                        help='do not gain wav.(default enabled)')
    parser.set_defaults(normalize=True)
    parser.add_argument('--gain_min',
                        type=int,
                        default=-20,
                        help='Gain min(db).')
    parser.add_argument('--gain_max',
                        type=int,
                        default=0,
                        help='Gain max(db).')
    # 变速
    parser.add_argument('--tempo',
                        dest='tempo',
                        action='store_true',
                        help='change speed (default enabled)')
    parser.add_argument('--no-tempo',
                        dest='tempo',
                        action='store_false',
                        help='do not change speed (default enabled)')
    parser.set_defaults(tempo=True)
    parser.add_argument('--speed_min',
                        type=float,
                        help='Speed min.',
                        default=0.9)
    parser.add_argument('--speed_max',
                        type=float,
                        help='Speed max.',
                        default=1.1)
    # 变调
    parser.add_argument('--pitch',
                        dest='pitch',
                        action='store_true',
                        help='enable change pitch.(default enabled)')
    parser.add_argument('--no-pitch',
                        dest='pitch',
                        action='store_false',
                        help='disable change pitch.(default enabled)')
    parser.set_defaults(pitch=True)
    parser.add_argument('--shift_min',
                        type=int,
                        help='pitch shift min.',
                        default=-5)
    parser.add_argument('--shift_max',
                        type=int,
                        help='pitch shift max.',
                        default=5)
    # 加混响（算法模拟混响，好像不需要混响数据集）
    parser.add_argument('--reverb',
                        dest='reverb',
                        action='store_true',
                        help='enable reverb.(default enabled)')
    parser.add_argument('--no-reverb',
                        dest='reverb',
                        action='store_false',
                        help='disable reverb.(default enabled)')
    parser.set_defaults(reverb=True)
    parser.add_argument('--reverberance_min',
                        type=int,
                        help='reverb strength.',
                        default=20)
    parser.add_argument('--reverberance_max',
                        type=int,
                        help='reverb strength.',
                        default=40)        


    parser.add_argument('--noise_transcript',
                        type=str,
                        help='noise wav transcript file.')
    parser.add_argument('--repeat', type=int, default=1, help='repeat times')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of process workers.')

    args = parser.parse_args()

    def find_option_type(key, parser):
        # print(dir(parser))
        for opt in parser._get_optional_actions():
            # print(opt)
            if ('--' + key) in opt.option_strings:
                return opt.type

    if args.config is not None:
        with open(args.config, 'r') as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k,v in yml_config.items():
            if k in args.__dict__:
                typ = find_option_type(k, parser)
                args.__dict__[k] = typ(v)
            else:
                sys.stdeer.write('Ignore unknow parameter {}.\n'.format(k))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    raw_utts = []
    with open(args.input_transcript, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            if '\t' in line:
                wav_path, text = line.strip().split('\t')
            else:
                wav_path, text = line.strip().split(' ', 1)
            raw_utts.append((wav_path, text))
    
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
                        wav_path, text = line.strip().split('\t', 1)
                    else:
                        wav_path, text = line.strip().split(' ', 1)
                    noise_utts.append((wav_path, text))

    all_perturbed_utts = []
    for i in range(args.repeat):
        with ProcessPoolExecutor(max_workers=args.num_workers) as e:
            all_perturbed_utts += e.map(functools.partial(
                _perturb_utt, args=args, noise_utts=noise_utts),
                                        raw_utts,
                                        chunksize=32)

    utts = [u for u in all_perturbed_utts if u is not None]
    # if shuffle:
    #     utts = utts.copy()
    #     random.shuffle(utts)
    with open(args.output_transcript, 'w', encoding='utf-8') as f:
        for wav, text in utts:
            f.write('{}\t{}\n'.format(wav, text))
