TCN唤醒任务训练步骤：

1. 检查训练数据文件名是否合法（文件名没有空格、以及其他一些需要转义的字符），非法的文件名在shell脚本中经常会出现错误；
如果有利用scripts/rename_wav_from_dir.py脚本进行批量重命名；

2. 使用脚本生成transcript，使用scripts/mk_trans_from_dir.py脚本；由于数据的标注形式各异，因此具体如何生成标注需要自行修改代码。此处举例，若标注在音频名中某一固定字段，
若负样本的标注未知，统一用garbage代替；运行以下命令会在data_dir（存放原始数据的目录）目录下生成trans.txt
python scripts/mk_trans_from_dir.py ${data_dir}

3. 检查训练数据的格式（要求采样率16000、16位精度、单通道）；
python scripts/check_wav_format.py ${data_dir}/trans.txt

4. 若数据格式不统一，可用以下脚本进行转换：（转换完之后需要把normed_dir里面数据全部替换到data_dir下面）
python scripts/norm_wav_from_trans.py ${data_dir}/trans.txt normed_dir

5. 将音频数据按5秒长度切分，主要是处理较长的负样本，防止训练时占用很大显存：
python scripts/split_duration_from_trans.py \
    ${data_dir}/trans_neg.txt ${data_dir}_split_neg

6. 检查训练数据的长度（要求在1.5s到5s之间）（有问题的数据需要人工审核再修改替换到data_dir里面）：
python scripts/check_wav_duration.py ${data_dir}/trans.txt

7. 检查训练数据是否为空、首尾静音段长度是否满足要求；不符合要求的音频会拷贝一份到outdir（有问题的数据需要人工审核再修改替换到data_dir里面）
python scripts/check_wav_silence.py ${data_dir}/trans.txt outdir

8. 利用ASR对正样本中的关键词进行边界划定，生成新的trans.txt_frame.trans：
/media/hdd2/kai.zhou/wenet/for_hotword/prepare_frame_label_using_asr.sh \
    ${data_dir}/trans.txt  CN

9. 检查正样本中的关键词的持续时长，理论上应该要小于模型的感受野，不合格的数据修正或去除：
python scripts/check_kws_duration.py ${data_dir}/trans.txt_frame.trans outdir

10. 分层切分数据集，将数据集以不同类别按比例切分，生成训练集和测试集
python scripts/split_train_dev_test_from_trans.py \
    --trans ${data_dir}/trans.txt \
    --train ${data_dir}/train_wav.trans \
    --dev ${data_dir}/dev_wav.trans \
    --test ${data_dir}/test_wav.trans \
    --words "嗨小问" \ #多个词用英文逗号隔开，例："播放音乐,停止播放,上一首,下一首"
    --rate 0.1

11. 对数据做增强（音量调整、语速调整、声调调整、加噪、混响、脉冲响应），需要噪声数据集、真实脉冲响应数据集：（还可以加上编解码、翻录），数据增强对模型的提升非常大，特别是训练数据较少的情况。
aug_outdir=${data_dir}_aug
rir_trans=/ssd1/kai.zhou/corpus/noise/rir/RIR_database/SLR26/simulated_rirs_48k/trans.txt
noise_trans=/media/hhd1/corpus/musan/noise/musan_split5s/trans.txt
python scripts/audio_augment.py \
    --input_transcript ${data_dir}/train_wav.trans \
    --output_dir ${aug_outdir}_noise \
    --gain_prob 0.8 \
    --gain_min -5 \
    --gain_max 5 \
    --speed_prob 0.8 \
    --speed_min 0.9 \
    --speed_max 1.1 \
    --pitch_prob 0.8 \
    --pitch_min -5 \
    --pitch_max 5 \
    --reverb_prob 0 \
    --reverb_min 5 \
    --reverb_max 20 \
    --rir_prob 0.5 \
    --rir_transcript ${rir_trans} \
    --noise_prob 0.8 \
    --snr_min -8 \
    --snr_max 15 \
    --noise_transcript ${noise_trans} \
    --repeat 1 \
    --num_workers 24


12. 将所有的训练数据组合在一起作为训练集，将所有的验证数据组合在一起，测试数据组合在一起：
cat ${trans_dir}/train_wav.trans \
    ${aug_outdir}_noise/trans.txt \
    > ${exp_dir}/all_frame.trans_train

cat ${trans_dir}/dev_wav.trans \
    > ${exp_dir}/all_frame.trans_dev
    
cat ${trans_dir}/test_wav.trans \
    > ${exp_dir}/all_frame.trans_test

13. 提取特征，下面脚本使用GPU并行提取特征(不使用GPU也没有影响，速度瓶颈一般在磁盘的io上，cpu占用其实并不高)，并以batchsize的大小存在一个pt文件中，最后记录在pt.trans中：
python ${root}/prepare_feat_pt_from_trans.py \
    ${exp_dir}/all_frame.trans_train \
    ${exp_dir}/train_pt.trans \
    ${pt_dir}/train \
    1

python ${root}/prepare_feat_pt_from_trans.py \
    ${exp_dir}/all_frame.trans_dev \
    ${exp_dir}/dev_pt.trans \
    ${pt_dir}/dev \
    1
   
python ${root}/prepare_feat_pt_from_trans.py \
    ${exp_dir}/all_frame.trans_test \
    ${exp_dir}/test_pt.trans \
    ${pt_dir}/test \
    1

14. 计算cmvn
python ${root}/compute_cmvn.py \
        ${exp_dir}/train_pt.trans \
        ${exp_dir}/cmvn.json

15. 开始训练(这里的test分成正样本test_pt.trans、负样本neg_test_pt.trans，其中负样本需要自己找不包含关键词的数据，并按照上面脚本切割成不超过5s的短音频，然后按照上面提取特征)
python ${root}/train.py \
    --train_data ${exp_dir}/train_pt.trans \
    --val_data ${exp_dir}/dev_pt.trans \
    --test_pos_data ${exp_dir}/test_pt.trans \
    --test_neg_data ${exp_dir}/neg_test_pt.trans \
    --cmvn  ${exp_dir}/cmvn.json \
    --output_dir ${exp_dir}/scratch_tcn_rf217_hixiaowen \
    --hotwords  嗨小问 \
    --epochs 50  \
    --lr 0.1 \
    --batch_size 1024 \
    --use_gpu \
    --gpu_ids 1 \
    --has_boundary \
    --model_type tcn 

16. 模型测试
python ${root}/test_wav_trans.py \
    ${exp_dir}/scratch_dstcn5x64_rf126_xiaoqixiaoqi_bd_specaug/epoch79_258.pt \
    ${exp_dir}/cmvn.json \
    小祺小祺 \
    /ssd/nfs06/jiankang.wang/hotword_torch/exp/小祺小祺/20230517/test_wav/trans.txt


17. 分别把cmvn.json和epoch79_256.pt转成.h文件
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
