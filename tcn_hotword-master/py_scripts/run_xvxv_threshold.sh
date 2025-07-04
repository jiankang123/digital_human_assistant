#!/bin/bash

dir=$1

./py_scripts/run.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testP \
    小V小V  \
    ../torch/new_data/vivo_kws/exps/20221102/testP.trans

./py_scripts/run.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testN \
    小V小V  \
    ../torch/new_data/vivo_kws/exps/20221102/testN_valid.trans

./py_scripts/run.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testN2 \
    小V小V  \
    /media/hdd2/corpus/vivo_data/sbc_out/neg_1115_1117.trans 




