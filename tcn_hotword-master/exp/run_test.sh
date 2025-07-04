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
#分别把cmvn.json和epoch79_256.pt转成.h文件
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python ${root}/port/save_cmvn.py \
      --cmvn_json ${exp_dir}/cmvn.json \
      --date 20230519 \
      --project qcc \
      --name XIAOQIXIAOQI_20230519

  python ${root}/port/save_params_small_qcc.py \
      --pt_file ${exp_dir}/scratch_dstcn5x64_rf126_xiaoqixiaoqi_bd_specaug/epoch79_258.pt \
      --project m510_xiaoqixiaoqi \
      --date 20230519 \
      --name XIAOQIXIAOQI_20230519


fi

#根据trans(原音频)测试模型效果
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  python ${root}/test_wav_trans.py \
    ${exp_dir}/scratch_dstcn5x64_rf126_xiaoqixiaoqi_bd_specaug/epoch79_258.pt \
    ${exp_dir}/cmvn.json \
    小祺小祺 \
    /ssd/nfs06/jiankang.wang/hotword_torch/exp/小祺小祺/20230517/test_wav/trans.txt
 
fi


#根据trans(提取完c_fbank特征后)测试模型效果（可以根据不同阈值看出唤醒率和误唤醒效果）
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  bash ${root}/compute_pt_trans_threshold_withn.sh \
    ${exp_dir}/scratch_dstcn5x64_rf126_hixiaowen_bd_specaug/epoch78_70.pt \
    ${exp_dir}/cmvn.json \
    小祺小祺 \
    ${root}/exp/hixiaowen2/test_data_trans/trans_cn-celeb_pt.txt


  bash ${root}/compute_pt_trans_threshold_withn.sh \
    ${exp_dir}/scratch_dstcn5x64_rf126_hixiaowen_bd_specaug/epoch78_70.pt \
    ${exp_dir}/cmvn.json \
    小祺小祺 \
    ${root}/exp/hixiaowen2/test_data_trans/trans_android_wav_pt.txt

  bash ${root}/compute_pt_trans_threshold_withn.sh \
    ${exp_dir}/scratch_dstcn5x64_rf126_hixiaowen_bd_specaug/epoch78_70.pt \
    ${exp_dir}/cmvn.json \
    小祺小祺 \
    ${root}/exp/hixiaowen2/test_data_trans/trans_ios_wav_pt.txt
fi

#测试单个音频并画图
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    python ${root}/test_wav.py py ${exp_dir}/scratch_dstcn5x64_rf126_xiaoqixiaoqi_bd_specaug/epoch79_258.pt ${exp_dir}/cmvn.json ${exp_dir}/test_wav_phone/1_20230523-151745.wav
fi