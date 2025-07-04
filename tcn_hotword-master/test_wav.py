import sys, os
import torch
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from feat import get_CMVN
import matplotlib.style as mplstyle
mplstyle.use('fast')
matplotlib.use('agg')
# plt.rcParams["font.family"] = ["sans-serif"]
# plt.rcParams["font.sans-serif"] = ['SimHei']

np.set_printoptions(precision=4) 
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)


FEAT_TYPE = "feat_cpp"
if FEAT_TYPE == "feat_cpp":
    from feat_cpp.feature import FilterbankExtractor
    filterbank_extractor = FilterbankExtractor()
elif FEAT_TYPE == "feat_c":
    from feat import compute_fbank
else:
    print("ERROR feat type!")
    exit(1)
print(f"Using {FEAT_TYPE} as filterbank")

# @profile
def plot_result(wav_path, pred_result, wav_pad=False, output_dir=None, label=None):
    garbage_confidence = pred_result[:,0] - 1
    pred_seq = pred_result.argmax(-1).tolist()
    num_classes = pred_result.shape[-1]
    fig_w = len(pred_seq) // 10
    fig_w = min(max(fig_w, 10), 655)
    
    # fig = plt.figure(figsize=(fig_w, 5))
    fig = plt.figure(figsize=(40, 5))
    if label:
        fig.suptitle(f"label: {label}", fontproperties="SimHei")
    if wav_path.lower().endswith('.wav'):
        wav_data = np.memmap(wav_path, dtype='h', offset=44, mode='r')
    else:
        wav_data = np.memmap(wav_path, dtype='h', offset=0, mode='r')

    det_time = np.linspace(0, len(wav_data), len(pred_seq))
    x_trick = np.linspace(0, len(wav_data)//16000, len(pred_seq))
    ax = fig.add_subplot(111)
    ax.plot(wav_data)
    ax2 = ax.twinx()
    ax2.plot(det_time, pred_seq, '-r')
    ax2.plot(det_time, garbage_confidence, '-y')
    ax2.grid()
    ax.set_xlabel("time")
    ax.set_ylabel("sample")
    ax2.set_ylabel("class")
    ax2.axis([0, len(wav_data), -2, num_classes])
    plt.xticks(x_trick)
    if not output_dir:
        pic_path = 'fig_dir/fig_{}.png'.format(os.path.basename(wav_path))
    else:
        pic_path = os.path.join(output_dir, 'fig_{}.png'.format(os.path.basename(wav_path)))
    fig.savefig(pic_path)
    plt.clf()
    plt.close()

# @profile
def wav_test(wav_path, net, mean, scale, output_dir=None, plot=True):
    if FEAT_TYPE == "feat_c":
        fbank = compute_fbank(wav_path)
    elif FEAT_TYPE == "feat_cpp":
        fbank = filterbank_extractor.extract(wav_path)
        fbank = torch.tensor(fbank, dtype=torch.float32)
    else:
        print("ERROR feat type!")
        exit(1)
    
    fbank = (fbank - mean) * scale
    # print(fbank.shape)
    # print(fbank)
    pred = net(fbank.unsqueeze(0)).squeeze(0).detach().numpy()
    # print(pred.shape)
    # print(np.around(pred, 4))
    if plot:
        plot_result(wav_path, pred, output_dir)
        

# @profile
def wav_simulate_test(wav_path, net, mean, scale, output_dir=None, plot=True):
    '''
    仿真c代码进行流式推理测试
    '''
    def full_connect(w, b, vec):
        return torch.matmul(w, vec) + b
    
    def depthwise_conv(w, mat):
        return (w * mat).sum(dim=0)
        
    
    def streaming_infer(feat):
        layer0_out = full_connect(layer0_w, layer0_b, feat)
        layer0_out = F.relu(layer0_out)
        
        sep0_input[indexs[0]] = layer0_out.T
        indexs[0] += 1
        sep0_out1 = depthwise_conv(sep0_w1, sep0_input[indexs[0]-8: indexs[0]])
        sep0_out2 = full_connect(sep0_w2, sep0_b, sep0_out1.reshape(-1, 1))
        sep0_out = F.relu(sep0_out2)
        
        if i % 4 != 0:
            return
         
        sep1_input[indexs[1]] = sep0_out.T
        indexs[1] += 1
        sep1_out1 = depthwise_conv(sep1_w1, sep1_input[indexs[1]-8: indexs[1]])
        sep1_out2 = full_connect(sep1_w2, sep1_b, sep1_out1.reshape(-1, 1))
        sep1_out = F.relu(sep1_out2)
        
        sep2_input[indexs[2]] = sep1_out.T
        indexs[2] += 1
        sep2_out1 = depthwise_conv(sep2_w1, sep2_input[indexs[2]-8: indexs[2]])
        sep2_out2 = full_connect(sep2_w2, sep2_b, sep2_out1.reshape(-1, 1))
        sep2_out = F.relu(sep2_out2)
        
        if i % 8 != 0:
            return 
        
        sep3_input[indexs[3]] = sep2_out.T
        indexs[3] += 1
        sep3_out1 = depthwise_conv(sep3_w1, sep3_input[indexs[3]-8: indexs[3]])
        sep3_out2 = full_connect(sep3_w2, sep3_b, sep3_out1.reshape(-1, 1))
        sep3_out = F.relu(sep3_out2)
        
        dense_out = full_connect(dense_w, dense_b, sep3_out.reshape(-1, 1))
        
        return F.softmax(dense_out.T, -1)
        
         
    
    fbank = compute_fbank(wav_path)
    layer0_w = net.layer0.weight.squeeze(-1)                # (64, 23)
    layer0_b = net.layer0.bias.reshape(-1, 1)               # (64, 1)
    sep0_w1 = net.SepConv1ds[0].depthwise_conv.weight.squeeze(1).T
    sep0_w2 = net.SepConv1ds[0].pointwise_conv.weight.squeeze(-1)
    sep0_b = net.SepConv1ds[0].pointwise_conv.bias.reshape(-1, 1)
    sep1_w1 = net.SepConv1ds[1].depthwise_conv.weight.squeeze(1).T
    sep1_w2 = net.SepConv1ds[1].pointwise_conv.weight.squeeze(-1)
    sep1_b = net.SepConv1ds[1].pointwise_conv.bias.reshape(-1, 1)
    sep2_w1 = net.SepConv1ds[2].depthwise_conv.weight.squeeze(1).T
    sep2_w2 = net.SepConv1ds[2].pointwise_conv.weight.squeeze(-1)
    sep2_b = net.SepConv1ds[2].pointwise_conv.bias.reshape(-1, 1)
    sep3_w1 = net.SepConv1ds[3].depthwise_conv.weight.squeeze(1).T
    sep3_w2 = net.SepConv1ds[3].pointwise_conv.weight.squeeze(-1)
    sep3_b = net.SepConv1ds[3].pointwise_conv.bias.reshape(-1, 1)
    dense_w = net.dense.weight.squeeze(-1) 
    dense_b = net.dense.bias.reshape(-1, 1) 
    
    sep0_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float)
    sep1_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float)
    sep2_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float)
    sep3_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float)
    indexs = [7] * 4
    
    fbank = (fbank - mean) * scale
    pred = []
    for i in range(0, fbank.shape[0], 2):
        result = streaming_infer(fbank[i].reshape(-1, 1))
        if result is None:
            
            continue
        else:
            detect = result.argmax()
            pred.append(result.detach().numpy())
            if detect != 0:
                print(detect, i)

    if plot:
        np_pred = np.array(pred)
        plot_result(wav_path, np_pred, output_dir)

# @profile
def wav_simulate_test_TCN(wav_path, net, mean, scale, output_dir=None, plot=True):
    '''
    仿真c代码进行流式推理测试
    '''
    def full_connect(w, b, vec):
        return torch.matmul(w, vec) + b

    conv0_w = net.convs[0].conv.weight.permute(0, 2, 1).reshape(64, -1)
    conv0_b = net.convs[0].conv.bias.reshape(-1, 1)
    conv1_w = net.convs[1].conv.weight.permute(0, 2, 1).reshape(64, -1)
    conv1_b = net.convs[1].conv.bias.reshape(-1, 1)
    conv2_w = net.convs[2].conv.weight.permute(0, 2, 1).reshape(64, -1)
    conv2_b = net.convs[2].conv.bias.reshape(-1, 1)
    conv3_w = net.convs[3].conv.weight.permute(0, 2, 1).reshape(64, -1)
    conv3_b = net.convs[3].conv.bias.reshape(-1, 1)
    conv4_w = net.convs[4].conv.weight.permute(0, 2, 1).reshape(64, -1)
    conv4_b = net.convs[4].conv.bias.reshape(-1, 1)
    dense_w = net.dense.weight
    dense_b = net.dense.bias.reshape(-1, 1)

    fbank = compute_fbank(wav_path)
    conv1_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float32)
    conv2_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float32)
    conv3_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float32)
    conv4_input = torch.zeros((fbank.shape[0], 64), dtype=torch.float32)
    indexs = [7] * 4

    def get_in(buf, pos, dilation):
        
        res = []
        pos_tmp = pos - dilation * 7 - 1
        for i in range(8):
            if pos_tmp < 0:
                res.append(torch.zeros(1, 64))
            else:
                res.append(buf[pos_tmp].reshape(1, -1))
            pos_tmp = pos_tmp + dilation
        return torch.concat(res)



    def streaming_infer(feat):
        conv0_out = full_connect(conv0_w, conv0_b, feat)
        conv0_out = F.relu(conv0_out)
        conv1_input[indexs[0]] = conv0_out.T
        indexs[0] += 1

        conv1_in = get_in(conv1_input, indexs[0], 2)
        conv1_out = full_connect(conv1_w, conv1_b, conv1_in.reshape(-1, 1))
        conv1_out = F.relu(conv1_out)
        conv2_input[indexs[1]] = conv1_out.T
        indexs[1] += 1

        conv2_in = get_in(conv2_input, indexs[1], 4)
        conv2_out = full_connect(conv2_w, conv2_b, conv2_in.reshape(-1, 1))
        conv2_out = F.relu(conv2_out)
        conv3_input[indexs[2]] = conv2_out.T
        indexs[2] += 1

        conv3_in = get_in(conv3_input, indexs[2], 4)
        conv3_out = full_connect(conv3_w, conv3_b, conv3_in.reshape(-1, 1))
        conv3_out = F.relu(conv3_out)
        conv4_input[indexs[3]] = conv3_out.T
        indexs[3] += 1

        conv4_in = get_in(conv4_input, indexs[3], 8)
        conv4_out = full_connect(conv4_w, conv4_b, conv4_in.reshape(-1, 1))
        conv4_out = F.relu(conv4_out)
        
        dense_out = full_connect(dense_w, dense_b, conv4_out.reshape(-1, 1))
        return F.softmax(dense_out.T, -1)
    
    fbank = (fbank - mean) * scale
    pred = []
    for i in range(8, fbank.shape[0]):
        result = streaming_infer(fbank[i-8: i].reshape(-1, 1))
        detect = result.argmax()
        pred.append(result.detach().numpy())
        if detect != 0:
            print(detect, i)

    if plot:
        np_pred = np.array(pred).squeeze()
        plot_result(wav_path, np_pred, output_dir)


# @profile
def main():
    test_mode = sys.argv[1]
    net_path = sys.argv[2]
    cmvn = sys.argv[3]
    wav_path = sys.argv[4]
    

    net = torch.load(net_path, map_location='cpu').eval()

    mean, scale = get_CMVN(cmvn)
    if test_mode == "py":
        wav_test(wav_path, net, mean, scale)
    elif test_mode == 'c':
        if hasattr(net, "bn0"):
            if hasattr(net.bn0, "running_mean"):
                merged_net = torch.quantization.fuse_modules(net,
                    [
                        ['layer0', 'bn0'],
                        ['SepConv1ds.0.pointwise_conv', 'SepConv1ds.0.bn'],
                        ['SepConv1ds.1.pointwise_conv', 'SepConv1ds.1.bn'],
                        ['SepConv1ds.2.pointwise_conv', 'SepConv1ds.2.bn'],
                        ['SepConv1ds.3.pointwise_conv', 'SepConv1ds.3.bn'],
                    ]
                )
                net = merged_net
        else:
            if hasattr(net.convs[0].bn, "running_mean"):
                remove_weight_norm(net.convs[0].conv)
                remove_weight_norm(net.convs[1].conv)
                remove_weight_norm(net.convs[2].conv)
                remove_weight_norm(net.convs[3].conv)
                remove_weight_norm(net.convs[4].conv)

                merged_net = torch.quantization.fuse_modules(net,
                    [
                        ['convs.0.conv', 'convs.0.bn'],
                        ['convs.1.conv', 'convs.1.bn'],
                        ['convs.2.conv', 'convs.2.bn'],
                        ['convs.3.conv', 'convs.3.bn'],
                        ['convs.4.conv', 'convs.4.bn'],
                    ]
                )
                net = merged_net
        if hasattr(net, "bn0"):
            wav_simulate_test(wav_path, net, mean, scale)
        else:
            wav_simulate_test_TCN(wav_path, net, mean, scale)


if __name__ == "__main__":
    main()