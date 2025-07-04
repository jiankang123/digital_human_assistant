#!/bin/bash

exe_path=$1
dir=$2
hotwords=$3
test_trans=$4

mkdir -p ${dir}

if [ -f ${test_trans}.list ]
then
    cp ${test_trans}.list ${dir}/test.list
else
    python py_scripts/prepare_wenet_data_from_trans.py \
        --trans  ${test_trans} \
        --datalist ${test_trans}.list
    cp ${test_trans}.list ${dir}/test.list
fi

python py_scripts/score_wav_trans.py \
    ${test_trans} \
    ${dir}/scores \
    ${exe_path}

python py_scripts/merge_score.py \
    ${dir}/scores

for key in 1 2 ; do
    python py_scripts/compute_det.py \
        ${key} \
        ${dir}/test.list \
        ${dir}/score.txt \
        ${dir}/stat_${key}.txt \
        ${hotwords}
done
