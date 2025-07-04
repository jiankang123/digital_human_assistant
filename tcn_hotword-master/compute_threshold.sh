#!/bin/bash


pt_file=$1
cmvn=$2
hotwords=$3
trans=$4


out_dir=${pt_file}${trans//\//_}
# echo ${out_dir}

if [ ! -f ${trans}.list ]
then
python /media/hdd2/kai.zhou/wenet/for_hotword/prepare_wenet_data_from_trans.py \
    --trans ${trans} \
    --datalist ${trans}.list
fi

if [ ! -f ${out_dir}/score.txt ]
then
python /ssd1/kai.zhou/workspace/hotword/torch/test_wav_trans.py \
    $pt_file \
    $cmvn \
    ${hotwords} \
    ${trans}
fi

for i in 1 ;do
    python /ssd1/kai.zhou/workspace/hotword/torch/compute_det_withn.py \
        ${i}  \
        ${trans}.list \
        ${out_dir}/score.txt \
        ${hotwords} \
        ${n}
done