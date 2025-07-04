import argparse
import sys
import os
import numpy as np
from multiprocessing import Process
import json
import torch
import shutil
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

def compute_acc(output_dir, copy=False, outdir=None):
    all_case = 0
    all_false = 0
    postive_case = 0
    postive_false = 0
    negative_case = 0
    negative_false = 0
    num_classes = 0
    results = []
    fr_file = open(os.path.join(output_dir, 'fr.list'), 'w', encoding='utf-8')
    fa_file = open(os.path.join(output_dir, 'fa.list'), 'w', encoding='utf-8')
    for line in open(os.path.join(output_dir, 'confidence.ark'), 'r', encoding='utf-8'):
        if '[' in line:
            fid, label, _ = line.strip().split(' ')
            if int(label) > num_classes:
                num_classes = int(label)
            score = []
        elif ']' in line:
            pred = np.array(score)
            pred_t = np.concatenate((pred[:,0].min().reshape(-1), 
                                     pred[:,1:].max(0)), 0)
            # 临时修改
            pred_class = pred_t.argmax()
            # if pred_t[1] >= 0.85:
            #     pred_class = 1
            # else:
            #     pred_class = 0
            #  

            if pred_class > num_classes:
                num_classes = pred_class
            results.append((pred_class, int(label)))
            all_case += 1
            if int(label) == 0:
                negative_case += 1
            else:
                postive_case += 1
            if pred_class != int(label):
                if int(label) == 0:
                    negative_false += 1
                    fa_file.write(fid + '\t' + str(pred_t[pred_class]) + '\n')
                else:
                    postive_false += 1
                    fr_file.write(fid + '\t' + str(pred_t[pred_class]) + '\n')
                print("{} recog error, label: {}, ".format(fid, label) + 
                      "predict: {}, confidence: {}".format(pred_class, 
                                                           pred_t[pred_class]))
                all_false += 1
                if copy and outdir:
                    outpath = os.path.join(outdir, os.path.basename(fid))
                    shutil.copy(fid, outpath)
        else:
            score.append([float(x) for x in line.split()])
    if postive_case == 0:
        FRR = -1
    else:
        FRR = postive_false/postive_case
    if negative_case == 0:
        FAR = -1
    else:
        FAR = negative_false/negative_case
    print("error rate: {:.4f}, FRR: {:.4f}, FAR: {:.4f}".format(
        (all_false/all_case), FRR, FAR)
    )
    fa_file.close()
    fr_file.close()


    num_classes += 1
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for p, t in results:
        matrix[p, t] += 1
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
        Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else -1.
        False_Alarm = round(FP / (FP + TN), 4) if FP + TN != 0 else -1.
        table2.add_row([i, Recall, False_Alarm])
    print(table2)


if __name__ == "__main__":
    dir = sys.argv[1]
    compute_acc(dir)
    
