
import sys, os
import torch
import torch.nn.functional as F
import numpy as np


def wav_simulate_test(wav_path, net, mean, scale, filterbank_extractor, output_dir=None, plot=True):
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
        
        return F.softmax(dense_out.T)
        
         
    
    fbank = filterbank_extractor.extract(wav_path)
    fbank = torch.tensor(fbank, dtype=torch.float32)
    
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
    for i in range(0, fbank.shape[0], 2):
        result = streaming_infer(fbank[i].reshape(-1, 1))
        if result is None:
            continue
        else:
            detect = result.argmax()
            if detect != 0:
                print(detect, i)


def wav_test(fbank, net, mean, scale, output_dir=None, plot=True):

    fbank = (fbank - mean) * scale
    pred = net(fbank.unsqueeze(0)).squeeze(0).detach().numpy()
    garbage_confidence = pred[:,0]
    seq = pred.argmax(-1)
    print(seq.tolist())
    
    num_class = 9
    if plot:
        from matplotlib import pyplot as plt
        fig = plt.figure()
        if wav_path.lower().endswith('.wav'):
            wav_data = np.memmap(wav_path, dtype='h', offset=44, mode='r')
        else:
            wav_data = np.memmap(wav_path, dtype='h', offset=0, mode='r')
        max_val = np.max(wav_data)

        det_time = np.linspace(0, len(wav_data), len(seq))
        ax = fig.add_subplot(111)
        ax.plot(wav_data)
        ax2 = ax.twinx()
        ax2.plot(det_time, seq, '-r')
        ax2.plot(det_time, garbage_confidence, '-y')
        ax2.grid()
        ax.set_xlabel("time")
        ax.set_ylabel("sample")
        ax2.set_ylabel("class")
        if not output_dir:
            pic_path = 'fig_dir/fig_{}.png'.format(os.path.basename(wav_path))
        else:
            pic_path = os.path.join(output_dir, 'fig_{}.png'.format(os.path.basename(wav_path)))
        fig.savefig(pic_path)
    

def main():
    wav_path = sys.argv[1]
    model = sys.argv[2]
    cmvn = sys.argv[3]
    
    net = torch.load(model, map_location='cpu').eval()
    try:
        net = net.module
    except:
        pass
    mean, scale = get_CMVN(cmvn)
    filterbank_extractor = FilterbankExtractor()
    wav_test(wav_path, net, mean, scale, filterbank_extractor)
    

    
def run_simulate():
    wav_path = sys.argv[1]
    model = sys.argv[2]
    cmvn = sys.argv[3]
    
    net = torch.load(model, map_location='cpu').eval()
    try:
        net = net.module
    except:
        pass
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
    mean, scale = get_CMVN(cmvn)
    filterbank_extractor = FilterbankExtractor()
    wav_simulate_test(wav_path, net, mean, scale, filterbank_extractor)
    
    
if __name__ == "__main__":
    # run_simulate()
    # main()
    print_feat(sys.argv[1])