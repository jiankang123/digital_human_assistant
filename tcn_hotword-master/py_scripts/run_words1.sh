#!/bin/bash

exe_path=$1
dir=$2
hotwords=$3
test_trans=$4

# exe_path=/ssd/nfs06/jiankang.wang/hotword_torch/py_scripts/bulid/hotword_qcc_model_infer
# dir=/ssd/nfs06/jiankang.wang/hotword_torch/py_scripts/feature
# hotwords=嗨小问,
# test_trans=/ssd/nfs06/jiankang.wang/hotword_torch/exp/hixiaowen2/test_data_trans/trans_cn-celeb_pt.txt

mkdir -p ${dir}

if [ -f ${test_trans}.list ]
then
    cp ${test_trans}.list ${dir}/test.list
else
    python prepare_wenet_data_from_trans.py \
        --trans  ${test_trans} \
        --datalist ${test_trans}.list
    cp ${test_trans}.list ${dir}/test.list
fi

python score_wav_trans.py \
    ${test_trans} \
    ${dir}/scores \
    ${exe_path}

python merge_score.py \
    ${dir}/scores

for key in 1; do
    python compute_det.py \
        ${key} \
        ${dir}/test.list \
        ${dir}/score.txt \
        ${dir}/stat_${key}.txt \
        ${hotwords}
done
