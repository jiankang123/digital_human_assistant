#!/usr/bin/env python3
# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: yongzou@mobvoi.com (Yong Zou), shenli@mobvoi.com (Shen Li)

import argparse
from concurrent.futures import ProcessPoolExecutor
import functools
import os
import random
import sox


def _perturb_utt(utt, args, noise_utts):
    raw_audio, label = utt
    raw_audio_filename = os.path.splitext(os.path.basename(raw_audio))[0]
    tns = sox.Transformer()
    surfix = ''
    if args.normalize:
        gain_db = random.randint(args.gain_min, args.gain_max)
        tns.gain(gain_db=gain_db)
        surfix += '_gain_' + str(gain_db)
    if args.tempo:
        speed = round(random.uniform(args.speed_min, args.speed_max), 3)
        tns.tempo(speed)
        surfix += '_tempo_' + str(speed)
    if args.pitch:
        shift = round(random.uniform(args.shift_min, args.shift_max), 3)
        tns.pitch(shift)
        surfix += '_pitch_' + str(shift)
    if args.reverb:
        tns.reverb()
        surfix += '_reverb'
    perturbed_audio = os.path.join(args.output_dir,
                                   raw_audio_filename + surfix + '.wav')
    if os.path.isfile(perturbed_audio):
        return None
    tns.build(raw_audio, perturbed_audio)

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
        tns_noise.trim(start_location, start_location + ori_duration)
        if args.normalize:
            # If normalize orginal audio, noise should be smaller than orginal file.
            # noise_gain_max = gain_db - 10
            noise_gain_max = gain_db + 10
            noise_gain_min = gain_db - 20
        else:
            # Otherwish set a reasonable default range
            noise_gain_max = -20
            noise_gain_min = -30
        gain_db = random.randint(noise_gain_min, noise_gain_max)
        tns_noise.gain(gain_db=gain_db)
        noise_trim_audio = os.path.join(
            args.output_dir, raw_audio_filename + ' _noise_trim.wav')
        tns_noise.build(noise_audio, noise_trim_audio)
        surfix += '_noise_' + os.path.splitext(
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
    parser.add_argument('--input_transcript',
                        help='Input transcript file.',
                        required=True)
    parser.add_argument('--output_dir',
                        help='Directory to save generated wav file.',
                        required=True)
    parser.add_argument('--output_transcript',
                        help='Output transcript file.',
                        required=True)
    parser.add_argument('--normalize',
                        dest='normalize',
                        action='store_true',
                        help='enable normalize gain.(default enabled)')
    parser.add_argument('--no-normalize',
                        dest='normalize',
                        action='store_false',
                        help='disable normalize gain.(default enabled)')
    parser.set_defaults(normalize=True)
    parser.add_argument('--gain_min',
                        type=int,
                        default=-20,
                        help='Gain min(db).')
    parser.add_argument('--gain_max',
                        type=int,
                        default=0,
                        help='Gain max(db).')
    parser.add_argument('--tempo',
                        dest='tempo',
                        action='store_true',
                        help='enable change tempo.(default enabled)')
    parser.add_argument('--no-tempo',
                        dest='tempo',
                        action='store_false',
                        help='disable change tempo.(default enabled)')
    parser.set_defaults(tempo=True)
    parser.add_argument('--speed_min',
                        type=float,
                        help='Speed min.',
                        default=0.9)
    parser.add_argument('--speed_max',
                        type=float,
                        help='Speed max.',
                        default=1.1)
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
    parser.add_argument('--reverb',
                        dest='reverb',
                        action='store_true',
                        help='enable reverb.(default enabled)')
    parser.add_argument('--no-reverb',
                        dest='reverb',
                        action='store_false',
                        help='disable reverb.(default enabled)')
    parser.set_defaults(reverb=True)
    parser.add_argument('--noise_transcript',
                        help='noise wav transcript file.')
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
                wav_path, text = line.strip().split('\t')
            else:
                wav_path, text = line.strip().split(' ', 1)
            raw_utts.append((wav_path, text))
    
    noise_utts = []
    if args.noise_transcript:
        with open(args.noise_transcript, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                if '\t' in line:
                    wav_path, text = line.strip().split('\t')
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
            f.write('{} {}\n'.format(wav, text))
