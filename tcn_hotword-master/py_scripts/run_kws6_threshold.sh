#!/bin/bash

dir=$1

./py_scripts/run_kws6.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testP \
    接听电话,拒接电话,上一首,下一首,调大音量,调小音量  \
    ../torch/new_data/vivo_kws/exps/20221201/testP.trans

./py_scripts/run_kws6.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testN \
    接听电话,拒接电话,上一首,下一首,调大音量,调小音量  \
    ../torch/new_data/vivo_kws/exps/20221102/testN_valid.trans

./py_scripts/run_kws6.sh \
    ./build/hotword_qcc_model_infer \
    ${dir}/testN2 \
    接听电话,拒接电话,上一首,下一首,调大音量,调小音量  \
    /media/hdd2/corpus/vivo_data/sbc_out/neg_1115_1117.trans 


rm -r ${dir}/test*/scores

