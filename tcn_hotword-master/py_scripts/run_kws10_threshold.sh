#!/bin/bash

dir=$1

./py_scripts/run_kws10.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testP \
    播放音乐,暂停播放,接听电话,拒接电话,调大音量,调小音量,开启降噪,关闭降噪,开启通透,关闭通透  \
    ../torch/new_data/vivo_kws/exps/20221102/testP.trans

./py_scripts/run_kws10.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testN \
    播放音乐,暂停播放,接听电话,拒接电话,调大音量,调小音量,开启降噪,关闭降噪,开启通透,关闭通透  \
    ../torch/new_data/vivo_kws/exps/20221102/testN_valid.trans

./py_scripts/run_kws10.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testN2 \
    播放音乐,暂停播放,接听电话,拒接电话,调大音量,调小音量,开启降噪,关闭降噪,开启通透,关闭通透  \
    /media/hdd2/corpus/vivo_data/sbc_out/neg_1115_1117.trans 


rm -r ${dir}/test*/scores

