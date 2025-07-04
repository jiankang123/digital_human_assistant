#!/bin/bash

#dir=$1

./run_words1.sh \
    /ssd/nfs06/jiankang.wang/hotword_torch/py_scripts/bulid/hotword_qcc_model_infer \
    /ssd/nfs06/jiankang.wang/hotword_torch/py_scripts/feature_heysiri \
    HEYSIRI  \
    /ssd/nfs06/jiankang.wang/hotword_torch/exp/heysiri/20230522/pos_test_wav.trans


#rm -r ${dir}/test*/scores