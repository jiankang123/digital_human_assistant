#!/bin/bash
set -e
stage=$1
stop_stage=$2
date=20230525
name=小祺小祺

root=/media/hdd2/jiankang.wang/works/cursor_agent/tcn_hotword-master

data_dir=/export/share-nfs-1/jiankang.wang/小祺小祺
exp_dir=${root}/exp/${name}/${date}
pt_dir=/ssd/nfs06/jiankang.wang/features/${name}/${date}
pos_data_dir=${data_dir}/pos_wav_dir
wav_data_trans=${data_dir}/wav_data_trans
tts_data=/export/share-nfs-1/jiankang.wang/小祺小祺/tts_data_wav_dir
tts_data_en=/home/jiankang.wang/tts_data_en
tts_data_trans=${data_dir}/tts_trans


mkdir -p ${exp_dir}
mkdir -p ${pt_dir}

test_pos_wav=${exp_dir}/pos_test_wav.trans
test_neg_wav=${exp_dir}/neg_test_wav.trans


# make wav_trans from data_dir
if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
   python ${root}/scripts/mk_trans_from_dir.py ${pos_data_dir}
fi


# check wav format
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
   python ${root}/scripts/check_wav_format.py ${pos_data_dir}/trans.txt
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   #python ${root}/scripts/check_wav_duration.py ${goole_data}/trans.txt outdir1
   python ${root}/scripts/check_wav_duration.py ${pos_data_dir}/trans.txt outdir1
      outdir=outdir1
   if [ -z "$(ls -A ${outdir})" ]; then
      echo "There are 0 files in outdir"
   else
      num_files=$(ls -1q $outdir | wc -l)
      echo "There are $num_files files in outdir"
   fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
# check wav silence
   python ${root}/scripts/check_wav_silence.py ${pos_data_dir}/trans.txt outdir2
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
# prepare frame label using asr
   /media/hdd2/kai.zhou/wenet/for_hotword/prepare_frame_label_using_asr.sh \
      ${pos_data_dir}/trans.txt  CN
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
# check kws duration from frame_trans
   python ${root}/scripts/check_kws_duration.py ${pos_data_dir}/trans.txt_frame.trans outdir
#problem: pos.trans_frame.trans_kws_duration_problem
#delete pos.trans_frame.trans_kws_duration_problem and create pos.trans neg.trans again
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

    python ${root}/scripts/split_train_dev_test_from_trans.py  \
        --trans ${pos_data_dir}/trans.txt_frame.trans \
        --train ${wav_data_trans}/pos/pos_train_trans.txt \
        --dev ${wav_data_trans}/pos/pos_dev_trans.txt \
        --test ${wav_data_trans}/pos/pos_test_trans.txt \
        --words "小祺小祺" \
        --rate 0.1

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
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

aug_outdir=${data_dir}/aug_outdir
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Merge data, start at ${now}"

    cat ${wav_data_trans}/pos/pos_train_trans.txt \
        ${aug_outdir}/pos_train_xiaoqixiaoqi_music/trans.txt \
        ${aug_outdir}/pos_train_xiaoqixiaoqi_noise/trans.txt \
        ${aug_outdir}/pos_train_xiaoqixiaoqi_talk/trans.txt \
        ${root}/exp/小祺小祺/20230517/neg_old_data/train_neg_487p_60000_aug_wav.trans \
        ${root}/exp/小祺小祺/20230517/neg_old_data/train_neg_cn-celeb2_60000_wav.trans \
        ${root}/exp/小祺小祺/20230517/neg_old_data/train_neg_wenetspeech_podcast_60000_wav.trans \
        ${root}/exp/小祺小祺/20230517/neg_old_data/train_neg_wenetspeech_youtube_60000_wav.trans \
        > ${exp_dir}/train_wav.trans

    cat ${wav_data_trans}/pos/pos_dev_trans.txt \
        ${aug_outdir}/pos_dev_xiaoqixiaoqi_music/trans.txt \
        ${aug_outdir}/pos_dev_xiaoqixiaoqi_noise/trans.txt \
        ${aug_outdir}/pos_dev_xiaoqixiaoqi_talk/trans.txt \
        ${dev_neg_wav_trans} \
        > ${exp_dir}/dev_wav.trans

    cat ${wav_data_trans}/pos/pos_test_trans.txt \
        ${aug_outdir}/pos_test_xiaoqixiaoqi_music/trans.txt \
        ${aug_outdir}/pos_test_xiaoqixiaoqi_noise/trans.txt \
        ${aug_outdir}/pos_test_xiaoqixiaoqi_talk/trans.txt \
        > ${exp_dir}/pos_test_wav.trans


    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi