import os, sys, shutil
import multiprocessing as mlp
import subprocess
from tqdm import tqdm 
from prettytable import PrettyTable
import numpy as np

def confusion_matrix(num_classes, pred, label):
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    assert len(pred) == len(label)
    for i in range(len(pred)):
        matrix[pred[i], label[i]] += 1

    table = PrettyTable()
    table.title = "Confusion Matrix"
    table.field_names = ["Predict\Label"] + [str(x) for x in range(num_classes)]
    for i in range(num_classes):
        table.add_row([i] + matrix[i].tolist())
    print(table)

    table2 = PrettyTable()
    table2.title = "Recall & False_Alarm"
    table2.field_names = ["", "Recall", "False_Alarm"]
    for i in range(num_classes):
        TP = matrix[i, i]
        FP = np.sum(matrix[i, :]) - TP
        FN = np.sum(matrix[:, i]) - TP
        TN = np.sum(matrix) - TP - FP - FN
        Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else -1.
        False_Alarm = round(FP / (FP + TN), 3) if FP + TN != 0 else -1.
        table2.add_row([i, Recall, False_Alarm])
    print(table2)
# HOTWORDS = {
#     "小V小V": 1,
#     "HIJOVI": 2,
#     "播放音乐": 3,
#     "暂停播放": 4,
#     "接听电话": 5,
#     "拒接电话": 6,
#     "上一首": 7,
#     "下一首": 8,
#     "调大音量": 9,
#     "调小音量": 10,
#     "开启降噪": 11,
#     "关闭降噪": 12,
#     "开启通透": 13,
#     "关闭通透": 14
# }

def fun(cmd, label, wav_path):
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    lines = out.decode()
    results = []
    for line in lines.split('\n'):
        if 'cmd' in line:
            results.append(int(line.split('cmd')[1].split(' ')[1][:-1]))
    
    return (results, label, wav_path)



def main():
    exe_path = sys.argv[1]
    test_trans = sys.argv[3]
    hotwords = sys.argv[2].split(',')
    out_dir = '/tmp/test_result'
    hotword_dict = {"GARBAGE": 0}
    for index, hotword in enumerate(hotwords):
        hotword_dict[hotword] = index+1
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tasks = []
    pool = mlp.Pool(24)

    # exe_path = './build/hotword_qcc_model_infer'

    with open(test_trans, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    for i in range(len(lines)):
        line = lines[i]
        wav_path = line.split('\t')[0]
        text = line.split('\t')[1]
        text = text.upper()
        if text in hotword_dict:
            label = hotword_dict[text]
        else:
            flag = False
            for k in hotword_dict.keys():
                if k in text:
                    flag = True
                    break
            if flag:
                label = hotword_dict[k]
            else:
                label = 0
        score_path = os.path.join(out_dir, os.path.basename(wav_path)[:-4]+'.txt')
        cmd = '{} {} > {}'.format(exe_path, wav_path, score_path)
        # subprocess.check_output(cmd, shell=True, stderr=subprocess.PIPE)
        tasks.append(pool.apply_async(fun, (cmd, label, wav_path)))

    results = []
    for i in tqdm(range(len(tasks))):
        results.append(tasks[i].get())

    positive_correct = 0
    positive_num = 0
    negative_correct = 0
    negative_num = 0

    labels = []
    preds = []
    for result, label, wav_path in results:
        labels.append(label)
        if label != 0:
            positive_num += 1
            if label in result:
                positive_correct += 1
                preds.append(label)
            else:
                if len(result) > 0:
                    preds.append(result[0])
                else:
                    preds.append(0)
                rel = []
                for k in result:
                    rel.append(list(hotword_dict.keys())[k])
                print(wav_path+'\t'+ ' '.join(rel))
        else:
            negative_num += 1
            if len(result) == 0:
                negative_correct += 1
                preds.append(label)
            else:
                preds.append(result[0])
                rel = []
                for k in result:
                    rel.append(list(hotword_dict.keys())[k])
                print(wav_path+'\t'+ ' '.join(rel))
    if negative_num != 0:
        FAR = (negative_num - negative_correct) / negative_num
    else:
        FAR = -1
    
    if positive_num != 0:
        FRR = (positive_num - positive_correct) / positive_num
    else:
        FRR = -1



    print(" FRR: {:.4f}, FAR: {:.4f}".format(FRR, FAR))
    confusion_matrix(len(hotwords)+1, preds, labels)
    shutil.rmtree(out_dir)



if __name__ == "__main__":
    main()
    # cmd = "./build/hotword_qcc_model_infer /media/hdd2/corpus/vivo_new_data/问问debug/误唤醒/wav_12/20221012-200759_R_环境噪音_jovi.wav >k"
    # fun(cmd, 0)