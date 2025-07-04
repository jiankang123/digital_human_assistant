import sys, os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from prettytable import PrettyTable

from dataloader import kws_dataset
from feat import get_CMVN
from test_wav import plot_result


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

    Recalls = []
    False_Alarms = []
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
        Recalls.append(Recall)
        False_Alarms.append(False_Alarm)
    print(table2)
    return Recalls, False_Alarms




def frame2wav_pred(pred, length):
    results = []
    num_samples = pred.shape[0]
    num_classes = pred.shape[-1]
    for i in range(num_samples):
        pred_l = [pred[i][:int(length[i].item()), 0].min()]
        for j in range(1, num_classes):
            pred_l.append(pred[i][:int(length[i].item()), j].max())
        pred_t = torch.stack(pred_l).T
        results.append(pred_t.argmax(-1).item())
    return results


# @profile
def main():
    net_path = sys.argv[1]
    cmvn = sys.argv[2]
    hotwords_str = sys.argv[3]
    trans_path = sys.argv[4]
    
    use_gpu = True

    out_dir = net_path + trans_path.replace("/", "_")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    score_file = open(os.path.join(out_dir, 'score.txt'), 'w', encoding='utf-8')

    # 将热词组织成字典，key为热词，value为类别
    hotwords_dict = {}
    hotwords = ["GARBAGE"] + hotwords_str.split(',')
    num_classes = len(hotwords)
    for index, hotword in enumerate(hotwords):
        if '/' in hotword:
            hotword_arr = hotword.split('/')
            for h in hotword_arr:
                hotwords_dict[h] = index
        else:
            hotwords_dict[hotword] = index
    print("hotwords:", hotwords_dict)

    net = torch.load(net_path, map_location='cpu').eval()
    mean, scale = get_CMVN(cmvn)
    if use_gpu:
        mean, scale, net = mean.cuda(), scale.cuda(), net.cuda()

    test_dataset = kws_dataset(trans_path, hotwords_dict, logging=False)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                num_workers=8, 
                                collate_fn=None, 
                                pin_memory=use_gpu,
                                drop_last=False,
    )
    size = 0
    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch_data in test_dataloader:
            X, l, y = batch_data
            X, l, y = X.squeeze(0), l.squeeze(0), y.squeeze(0)
            size += X.shape[0]
            if use_gpu:
                X, l = X.cuda(), l.cuda()
            X = (X - mean) * scale
            pred = net(X)
            if use_gpu:
                pred = pred.cpu()
            
            pred_results = frame2wav_pred(pred, l)
            labels = y.tolist()
            all_pred.extend(pred_results)
            all_label.extend(labels)

            torch_pred = torch.tensor(np.array(pred_results, dtype=np.int32))
            
            true_neg = torch.logical_and((torch_pred == 0), (y == 0)).type(torch.float).sum().item()
            all_neg = (y == 0).type(torch.float).sum().item()
            true_pos = torch.logical_and((torch_pred == y), (y != 0)).type(torch.float).sum().item()
            all_pos = (y != 0).type(torch.float).sum().item()
            if all_neg != 0:
                FAR = (all_neg - true_neg) / all_neg
            else:
                FAR = -1
            true_pos = torch.logical_and((torch_pred == y), (y != 0)).type(torch.float).sum().item()
            all_pos = (y != 0).type(torch.float).sum().item()
            if all_pos !=0:
                FRR = (all_pos - true_pos) / all_pos
            else:
                FRR = -1
            print(f"FRR: {(100*FRR):>0.4f} FAR: {(100*FAR):>0.4f}")

            # if FRR > 0.8:
            #     print(f"FRR: {(100*FRR):>0.4f} FAR: {(100*FAR):>0.4f}")
            # else:
            #     print(f"FRR: {(100*FRR):>0.4f} FAR: {(100*FAR):>0.4f}")
            for i in range(pred.shape[0]):
                for r in range(1, pred.shape[2]):
                    pred_tmp = ['%.4f'%x for x in pred[i][:int(l[i].item()), r]]
                    score_file.write(str(labels[i]) + ' ' + str(r) + ' '
                                        + ' '.join(pred_tmp)+'\n')
    score_file.close()
    confusion_matrix(num_classes, all_pred, all_label)


if __name__ == "__main__":
    main()