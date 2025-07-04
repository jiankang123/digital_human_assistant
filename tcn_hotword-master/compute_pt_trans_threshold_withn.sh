#!/bin/bash


pt_file=$1
cmvn=$2
hotwords=$3
trans=$4

out_dir=${pt_file}${trans//\//_}

if [ ! -f ${out_dir}/score.txt ];then
    echo "${out_dir}/score.txt does not exists, computing ..."
    python /ssd/nfs06/jiankang.wang/hotword_torch/test_pt_trans.py \
        $pt_file \
        $cmvn \
        ${hotwords} \
        ${trans}
else
    echo "${out_dir}/score.txt already exists, skip computation"
fi

for i in 1 ;do
    for n in 1 2 3 4 5 ;do
        python /ssd/nfs06/jiankang.wang/hotword_torch/compute_det_withn.py \
            ${i}  \
            ${out_dir}/score.txt \
            ${hotwords} \
            ${n} &
    done
done
wait