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
tts_data_trans=${data_dir}/tts_trans
wav_data_trans=${data_dir}/wav_data_trans
dev_neg_wav_trans=${root}/exp/${name}/dev_neg_487p_wav.trans

#mkdir -p ${exp_dir}
mkdir -p ${pt_dir}

train_pt=${exp_dir}/train_pt.trans
dev_pt=${exp_dir}/dev_pt.trans
test_pos_pt=${exp_dir}/test_pos_pt
test_neg_pt=${exp_dir}/test_neg_pt
test_pos_wav=${exp_dir}/pos_test_wav.trans
test_neg_wav=${exp_dir}/neg_test_wav.trans



if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Merge data, start at ${now}"

    python ${root}/prepare_feat_pt_from_trans.py \
        ${exp_dir}/train_wav.trans \
        ${pt_dir}/train/trans.txt \
        ${pt_dir}/train \
        1
    
    cp ${pt_dir}/train/trans.txt \
        ${train_pt}
    
    python ${root}/prepare_feat_pt_from_trans.py \
        ${exp_dir}/dev_wav.trans \
        ${pt_dir}/dev/trans.txt \
        ${pt_dir}/dev \
        1

    cp ${pt_dir}/dev/trans.txt \
        ${dev_pt} 

    python ${root}/prepare_feat_pt_from_trans.py \
        ${test1_pos_wav} \
        ${pt_dir}/test/pos/trans.txt \
        ${pt_dir}/test/pos \
        1

    cp ${pt_dir}/test/pos/trans.txt \
        ${test_pos_pt}

    cp ${test1_neg_pt} \
        ${test_neg_pt}


    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "Prepare cmvn, start at ${now}"
    # 计算cmvn
    python ${root}/compute_cmvn.py \
        ${train_pt} \
        ${exp_dir}/cmvn.json
    
    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "training, start at ${now}"

    python ${root}/train.py \
        --train_data ${train_pt} \
        --val_data ${dev_pt} \
        --test_pos_data ${test_pos_pt}  \
        --test_neg_data ${test_neg_pt} \
        --cmvn  ${exp_dir}/cmvn.json \
        --net_config config_dstcn5x64_rf126 \
        --avg \
        --use_specaug \
        --output_dir ${exp_dir}/scratch_dstcn5x64_rf126_xiaoqixiaoqi_bd_specaug \
        --hotwords  小祺小祺 \
        --epochs 80  \
        --lr 0.1 \
        --batch_size 1024 \
        --use_gpu \
        --gpu_ids 1 \
        --has_boundary

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "training, start at ${now}"

    python ${root}/train.py \
        --train_data ${train_pt} \
        --val_data ${dev_pt} \
        --test_pos_data ${test_pos_pt} \
        --test_neg_data ${test_neg_pt} \
        --cmvn  ${exp_dir}/cmvn.json \
        --net_config config_dstcn5x64_rf126 \
        --avg \
        --output_dir ${exp_dir}/scratch_dstcn5x64_rf126_xiaoqixiaoqi_bd \
        --hotwords  小祺小祺 \
        --epochs 80  \
        --lr 0.1 \
        --batch_size 1024 \
        --use_gpu \
        --gpu_ids 1 \
        --has_boundary

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "training, start at ${now}"
    python ${root}/train.py \
        --train_data ${train_pt} \
        --val_data ${dev_pt} \
        --test_pos_data ${test1_pt} \
        --test_neg_data ${test2_pt} \
        --cmvn  ${exp_dir}/cmvn.json \
        --net_config config_dstcn5x64_rf210 \
         --avg \
        --output_dir ${exp_dir}/scratch_dstcn5x64_rf210_xiaoqixiaoqi_bd \
        --hotwords  小祺小祺 \
        --epochs 80  \
        --lr 0.1 \
        --batch_size 1024 \
        --use_gpu \
        --gpu_ids 1 \
        --has_boundary

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "training, start at ${now}"
    python ${root}/train.py \
        --train_data ${train_pt} \
        --val_data ${dev_pt} \
        --test_pos_data ${test1_pt} \
        --test_neg_data ${test2_pt} \
        --cmvn  ${exp_dir}/cmvn.json \
        --use_specaug \
        --net_config config_dstcn5x64_rf210 \
         --avg \
        --use_specaug \
        --output_dir ${exp_dir}/scratch_dstcn5x64_rf210_xiaoqixiaoqi_bd_specaug \
        --hotwords  小祺小祺 \
        --epochs 80  \
        --lr 0.1 \
        --batch_size 1024 \
        --use_gpu \
        --gpu_ids 1 \
        --has_boundary

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "training, start at ${now}"
    python ${root}/train.py \
        --train_data ${train_pt} \
        --val_data ${dev_pt} \
        --test_pos_data ${test1_pt} \
        --test_neg_data ${test2_pt} \
        --cmvn  ${exp_dir}/cmvn.json \
        --net_config config_dstcn5x128_rf126 \
        --avg \
        --output_dir ${exp_dir}/scratch_dstcn5x128_rf126_xiaoqixiaoqi_bd \
        --hotwords  小祺小祺 \
        --epochs 80  \
        --lr 0.1 \
        --batch_size 1024 \
        --use_gpu \
        --gpu_ids 1 \
        --has_boundary

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    now=`date +'%Y-%m-%d %H:%M:%S'`
    start_time=$(date --date="$now" +%s)
    echo "training, start at ${now}"
    python ${root}/train.py \
        --train_data ${train_pt} \
        --val_data ${dev_pt} \
        --test_pos_data ${test1_pt} \
        --test_neg_data ${test2_pt} \
        --cmvn  ${exp_dir}/cmvn.json \
        --use_specaug \
        --net_config config_dstcn5x128_rf126 \
        --avg \
        --output_dir ${exp_dir}/scratch_dstcn5x128_rf126_xiaoqixiaoqi_bd_specaug \
        --hotwords  小祺小祺 \
        --epochs 80  \
        --lr 0.1 \
        --batch_size 1024 \
        --use_gpu \
        --gpu_ids 1 \
        --has_boundary

    now=`date +'%Y-%m-%d %H:%M:%S'`
    end_time=$(date --date="$now" +%s);
    echo "Done, used time "$((end_time-start_time))"s"
fi





