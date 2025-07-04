import sys, os
import torch
import time
import numpy as np
import multiprocessing as mlp
from tqdm import tqdm
from feat import get_CMVN
from anlaysis_confidence import compute_acc
from test_wav import plot_result
from dataloader import trans_test_dataset, trans_test_Collate_fn
from filterbank_matrix import compute_fbank_matrix
from feat_cpp.feature import compute_cpp_filterbank_batch


FEAT_TYPE = "cpp_fbank"
print(f"Using FEAT_TYPE: {FEAT_TYPE}")

def frame2wav_pred(pred):
    num_classes = pred.shape[1]
    pred_l = [torch.min(pred[:, 0], 1).values]
    for i in range(1, num_classes):
        pred_l.append(torch.max(pred[:, i], 1).values)
    pred_t = torch.stack(pred_l).T
    return pred_t.argmax()

# @profile
def main():
    """
    python trans_test.py 
    
    """
    # start_time = time.time()
    net_path = sys.argv[1]
    cmvn = sys.argv[2]
    hotwords_str = sys.argv[3]
    trans_path = sys.argv[4]
    
    plot = False
    if len(sys.argv) > 5:
        plot = sys.argv[5] == "plot"
    
    # use_gpu = True
    use_gpu =False

    
    # plot = False
    out_dir = net_path + trans_path.replace("/", "_")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if plot:
        fig_dir = os.path.join(out_dir, "figs_")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)


    # 将热词组织成字典，key为热词，value为类别
    hotwords_dict = {}
    hotwords = ["GARBAGE"] + hotwords_str.split(',')
    num_classes = len(hotwords)
    for index, hotword in enumerate(hotwords):
        if '/' in hotword:
            hotword_arr = hotword.split('/')
            for h in hotword_arr:
                hotwords_dict[h.upper()] = index
        else:
            hotwords_dict[hotword.upper()] = index
    print("hotwords:", hotwords_dict)

    confident_ark = open(os.path.join(out_dir, 'confidence.ark'), 'w', 
                         encoding='utf-8')
    score_file = open(os.path.join(out_dir, 'score.txt'), 'w', 
                         encoding='utf-8')
    net = torch.load(net_path, map_location='cpu').eval()
    mean, scale = get_CMVN(cmvn)
    if use_gpu:
        net = net.cuda()
        mean, scale = mean.cuda(), scale.cuda()
    test_dataset = trans_test_dataset(trans_path, hotwords_dict)
    dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=32, 
        collate_fn=trans_test_Collate_fn, 
        pin_memory=use_gpu, 
        drop_last=False
    )
    
    plot_tasks = []
    pool = mlp.Pool(24)
    with torch.no_grad():
        for index, (audio, lengths, label, path) in enumerate(tqdm(dataloader)):
            # compute fbank
            if FEAT_TYPE == "c_fbank":
                fbank = compute_fbank_matrix(audio, use_gpu=use_gpu)
            elif FEAT_TYPE == "cpp_fbank":
                fbank = compute_cpp_filterbank_batch(audio)
            else:
                print(f"ERROR FEAT_TYPE {FEAT_TYPE}")
                exit(1)
            # fbank计算函数，outshape: 512*padded_length*23
            batch_size, length, nmel = fbank.shape
            lengths = [int(torch.ceil(l/160)) for l in lengths]
            # CMVN
            fbank = (fbank - mean) * scale
            pred = net(fbank)
            if isinstance(pred,tuple):
                pred = pred[0]
            if use_gpu:
                pred = pred.cpu()
            pred = pred.detach().numpy()

            # 画图需要串行处理，所以对batch的每一条语音操作
            for wav_index in range(batch_size):
                pred_per_wav = pred[wav_index,:lengths[wav_index]]
                path_per_wav = path[wav_index]
                name_per_wav = os.path.basename(path_per_wav).replace('.wav', '')
                label_text = list(hotwords_dict.keys()
                                  )[int(label[wav_index].item())]
                if plot:
                    plot_tasks.append(pool.apply_async(
                        plot_result, 
                        (path_per_wav, pred_per_wav, 0, fig_dir, label_text))
                    )
                confident_ark.write(path_per_wav + ' ' 
                                   + str(int(label[wav_index].item())) + ' [\n')
                for data in pred_per_wav:
                    for confidence in data:
                        confident_ark.write('%.4f ' % confidence)
                    confident_ark.write('\n')
                confident_ark.write(']' + '\n')
                for r in range(1, pred_per_wav.shape[1]):
                    pred_tmp = ['%.4f'%x for x in pred_per_wav[:,r]]
                    score_file.write(name_per_wav + ' ' + str(r) + ' '
                                        + ' '.join(pred_tmp)+'\n')
    confident_ark.close()
    score_file.close()
    compute_acc(out_dir)
    if plot:
        print("ploting ...")
        for i in tqdm(range(len(plot_tasks))):
            plot_tasks[i].get()
        print(f"pictures are in {fig_dir}")
    print("done")
    end_time = time.time()


if __name__ == "__main__":
    main()
