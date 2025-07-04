#!/bin/bash
set -e
stage=$1
stop_stage=$2
date=20230525
name=小祺小祺

root=/ssd/nfs06/jiankang.wang/hotword_torch

data_dir=/export/share-nfs-1/jiankang.wang/小祺小祺
exp_dir=${root}/exp/${name}/${date}
pt_dir=/ssd/nfs06/jiankang.wang/features/${name}/${date}
pos_data_dir=${data_dir}/pos_wav_dir
tts_data=/export/share-nfs-1/jiankang.wang/小祺小祺/tts_data_wav_dir
tts_data_en=/home/jiankang.wang/tts_data_en
tts_data_trans=${data_dir}/tts_trans

mkdir -p ${exp_dir}
mkdir -p ${pt_dir}

test_pos_wav=${exp_dir}/pos_test_wav.trans
test_neg_wav=${exp_dir}/neg_test_wav.trans


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Merge data, start at ${now}"

    python ${root}/scripts/get_tts_data_pitch_speed_volume.py \
        ${tts_data_en}/tts_pos_en
    # python ${root}/scripts/get_tts_data_pitch_speed_volume.py \
    #     ${tts_data}/tts_neg_500speaker_data

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Merge data, start at ${now}"

    python ${root}/add_silence_for_audio_path.py \
        ${tts_data}/tts_pos_500speaker_data \
        ${tts_data}/tts_pos_500speaker_data_add_silence \
        小祺小祺

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Merge data, start at ${now}"

    python ${root}/scripts/norm_wav_from_trans.py \
        ${tts_data}/tts_pos_500speaker_data_add_silence/trans.txt \
        ${tts_data}/tts_pos_500speaker_data_add_silence_norm

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Merge data, start at ${now}"
    #生成带边界trans: ${tts_data_trans}/tts_train_wav.trans_frame.trans

    /ssd/nfs06/kai.zhou2221/workspace/wenet_forhotword/for_hotword/prepare_frame_label_using_asr.sh \
        ${tts_data}/tts_pos_500speaker_data_add_silence_norm/trans.txt  CN

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Merge data, start at ${now}"


    python ${root}/scripts/split_train_dev_from_trans.py  \
        --trans ${tts_data}/tts_pos_500speaker_data_add_silence_norm/trans.txt_frame.trans \
        --train ${tts_data_trans}/pos_tts/train_pos_tts_wav.trans \
        --dev ${tts_data_trans}/pos_tts/dev_pos_tts_wav.trans \
        --words "小祺小祺" \
        --rate 0.1


    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#Set variables
    rir_trans=/export/expts2/yan.zhang/corpus/reverb_audio/rir_database/RIR_database/SLR26/simulated_rirs_48k/trans.txt
    noise=/ssd/nfs06/kai.zhou2221/open_source_data/musan/noise-split5s/trans.txt
    music=/ssd/nfs06/kai.zhou2221/open_source_data/musan/music-split5s/trans.txt
    talk=/ssd/nfs06/kai.zhou2221/open_source_data/musan/speech-split5s/trans.txt
# Augment data
   for dataset in "train" "dev" "test"; do
        for aug_type in "noise" ;do 
               aug_data=${!aug_type}
               aug_outdir=${data_dir}/aug_outdir/pos_${dataset}_xiaoqixiaoqi_${aug_type}
               echo "$"
               python ${root}/scripts/audio_augment.py \
                  --input_transcript ${wav_data_trans}/pos/pos_${dataset}_trans.txt \
                  --output_dir ${aug_outdir} \
                  --gain_prob 0.8 \
                  --gain_min 5 \
                  --gain_max 15 \
                  --speed_prob 0.8 \
                  --speed_min 0.9 \
                  --speed_max 1.1 \
                  --pitch_prob 0.8 \
                  --pitch_min -5 \
                  --pitch_max 5 \
                  --reverb_prob 0 \
                  --reverb_min 5 \
                  --reverb_max 20 \
                  --rir_prob 0.5 \
                  --rir_transcript ${rir_trans} \
                  --noise_prob 0.8 \
                  --snr_min -15 \
                  --snr_max 15 \
                  --noise_transcript ${aug_data} \
                  --repeat 1 \
                  --num_workers 24
        done
   done

   for dataset in "train" "dev" "test"; do
        for aug_type in  "music" ;do 
               aug_data=${!aug_type}
               aug_outdir=${data_dir}/aug_outdir/pos_${dataset}_xiaoqixiaoqi_${aug_type}
               echo "$"
               python ${root}/scripts/audio_augment.py \
                  --input_transcript ${wav_data_trans}/pos/pos_${dataset}_trans.txt \
                  --output_dir ${aug_outdir} \
                  --gain_prob 0.8 \
                  --gain_min 5 \
                  --gain_max 15 \
                  --speed_prob 0.8 \
                  --speed_min 0.9 \
                  --speed_max 1.1 \
                  --pitch_prob 0.8 \
                  --pitch_min -5 \
                  --pitch_max 5 \
                  --reverb_prob 0 \
                  --reverb_min 5 \
                  --reverb_max 20 \
                  --rir_prob 0.5 \
                  --rir_transcript ${rir_trans} \
                  --noise_prob 0.8 \
                  --snr_min 3 \
                  --snr_max 15 \
                  --noise_transcript ${aug_data} \
                  --repeat 1 \
                  --num_workers 24
        done
   done

   for dataset in "train" "dev" "test"; do
        for aug_type in "talk";do 
               aug_data=${!aug_type}
               aug_outdir=${data_dir}/aug_outdir/pos_${dataset}_xiaoqixiaoqi_${aug_type}
               echo "$"
               python ${root}/scripts/audio_augment.py \
                  --input_transcript ${wav_data_trans}/pos/pos_${dataset}_trans.txt \
                  --output_dir ${aug_outdir} \
                  --gain_prob 0.8 \
                  --gain_min 5 \
                  --gain_max 15 \
                  --speed_prob 0.8 \
                  --speed_min 0.9 \
                  --speed_max 1.1 \
                  --pitch_prob 0.8 \
                  --pitch_min -5 \
                  --pitch_max 5 \
                  --reverb_prob 0 \
                  --reverb_min 5 \
                  --reverb_max 20 \
                  --rir_prob 0.5 \
                  --rir_transcript ${rir_trans} \
                  --noise_prob 0.8 \
                  --snr_min 5 \
                  --snr_max 15 \
                  --noise_transcript ${aug_data} \
                  --repeat 1 \
                  --num_workers 24
        done
   done
fi


