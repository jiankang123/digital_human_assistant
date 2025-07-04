import os, sys
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import kws_dataset
from net.tcn_sep import TCN_SEP
from net.tcn import TCN
from pooling import max_pooling, maxpool_mask, max_pooling_new
from tricks.spec_augmentation import specaug_torch
from tricks.mixup import mixup
from feat import get_CMVN


class my_loss(nn.Module):
    def __init__(self, ):
        super(my_loss, self).__init__()
        
    def forward(self, logits, labels):
        loss = 0
        num_classes = logits.shape[-1]
        if labels.dtype is not torch.float32:
            labels = F.one_hot(labels, num_classes=num_classes).float()

        logits = torch.clamp(logits, 1e-8, 1.0)
        logits2 = torch.clamp(1 - logits, 1e-8, 1.0)
        # print(logits, logits2)
        for i in range(num_classes):
            loss += -torch.mean(
                labels[:, i] * torch.log(logits[:, i]) +
                (1 - labels[:, i]) * torch.log(logits2[:, i])
            )
            # print(loss)
        return loss


def save_pt(net, epoch, correct, test_loss, top_corrects, output_dir):
    if correct > top_corrects[0][0]:
        top_corrects.append((correct, epoch, test_loss))
        ptfile = os.path.join(output_dir, "epoch{}.pt".format(epoch+1))
        torch.save(net, ptfile)
        top_corrects.sort()
        bad_net = os.path.join(output_dir, "epoch{}.pt".format(
            top_corrects[0][1]+1, top_corrects[0][0]))
        if os.path.exists(bad_net):
            os.remove(bad_net)
        top_corrects.pop(0)
        with open(os.path.join(output_dir, 'info.txt'), 'w', encoding='utf-8') as fout:
            for k in top_corrects:
                fout.write(str(k[1]) + '\t' + str(k[0]) + '\t' + str(k[2]) + '\n')
        # print(top_corrects)
    else:
        return


def append_feature(X, l, sustained_frames):
    num, max_len, dim = X.shape
    l = l
    pad_tensor = torch.zeros((num, sustained_frames, dim), dtype=X.dtype, device=X.device)
    new_X = torch.cat((X, pad_tensor), dim=1)
    for i in range(num):
        new_X[i][l[i]: l[i]+sustained_frames] = torch.tile(X[i][l[i]-1], [sustained_frames, 1])
    new_l = l + sustained_frames
    return new_X, new_l


def train(args):
    
    # 将热词组织成字典，key为热词，value为类别
    hotwords = {}
    num_classes = len(args.hotwords)
    for index, hotword in enumerate(args.hotwords):
        if '/' in hotword:
            hotword_arr = hotword.split('/')
            for h in hotword_arr:
                hotwords[h.upper()] = index
        else:
            hotwords[hotword.upper()] = index
    print("hotwords:", hotwords)
    
    # 设置
    save_pt_nums = args.epochs
    top_corrects = [(0,0,0)] * save_pt_nums 
    fuse_model = args.fuse_model
    tcn_sep = args.tcn_sep
    residual = args.residual
    use_gpu = args.use_gpu
    use_specaug = args.use_specaug
    use_mixup = args.use_mixup
    pretrained_model = args.pretrained_model
    sustained_frames = args.sustained_frames
    block_size = 1024

    tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)

    # 设置超参数
    learning_rate = args.lr
    batch_size = args.batch_size // block_size # real batch size is set in example block size
    epochs = args.epochs
    
    if use_gpu:
        num_gpus = torch.cuda.device_count()
        # batch_size *= num_gpus
    
    if use_specaug:
        speaug_param = {
            "time_stretch": {
                "prob": 0,
                "rate_low": 0.8,
                "rate_high": 1.2 
            },
            "time_masking": {
                "prob": 0.8,
                "max_duration": 20
            },
            "freq_masking": {
                "prob": 0.8,
                "max_duration": 3
            }
        }
    if use_mixup:
        mixup_alpha = 0.4
    
    
    # 数据准备
    train_dataset = kws_dataset(args.train_data, hotwords, has_boundary=args.has_boundary, logging=False)
    val_dataset = kws_dataset(args.val_data, hotwords, has_boundary=args.has_boundary, logging=False)
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=0, 
                                collate_fn=None, 
                                pin_memory=use_gpu,
                                drop_last=False,
    )
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=0, 
                                collate_fn=None, 
                                pin_memory=use_gpu,
                                drop_last=False
    )
    mean, scale = get_CMVN(args.cmvn)
    if use_gpu:
        mean, scale = mean.cuda(), scale.cuda()
        
    # 构造模型
    if pretrained_model:
        net = torch.load(pretrained_model, map_location="cpu")
    else:
        num_layers = 5
        channels = [128] * num_layers
        dilations = [1, 2, 4, 4, 8]
        if tcn_sep:
            kernels = [1] + [8] * (num_layers - 1)
            net = TCN_SEP(channels, kernels, dilations, num_classes, residual)
        else:
            kernels = [8] * num_layers
            net = TCN(channels, kernels, dilations, num_classes, residual)
    
    if use_gpu:
        gpu_ids = args.gpu_ids
        # net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        try: 
            net = net.module
        except:
            pass
        net = net.cuda()
    
    if fuse_model:
        print("fuse model")
        merged_net = torch.quantization.fuse_modules(net.eval(),
            [
                ['layer0', 'bn0'],
                ['SepConv1ds.0.pointwise_conv', 'SepConv1ds.0.bn'],
                ['SepConv1ds.1.pointwise_conv', 'SepConv1ds.1.bn'],
                ['SepConv1ds.2.pointwise_conv', 'SepConv1ds.2.bn'],
                ['SepConv1ds.3.pointwise_conv', 'SepConv1ds.3.bn'],
            ]
        )
        net = merged_net
        
        
    # Loss
    loss_fn = my_loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,  momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8, last_epoch=-1, verbose=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        threshold=0.01,
    )
    # 训练循环
    for epoch in range(epochs):
        # train loop
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "Epoch %d" % epoch, "LR %f" % lr)
        size = len(train_dataset)
        net = net.train()
        for batch, batch_data in enumerate(train_dataloader):    # X.shape: (1, bs, seq_len, feat_dim)
            if not args.has_boundary:
                X, l, y = batch_data
            else:
                X, l, y, begins, ends = batch_data

            X, l, y = X.squeeze(0), l.squeeze(0), y.squeeze(0)  # X.shape: (bs, seq_len, feat_dim)
            
            # append sustained_frames feature
            if args.append_feature:
                l = l.type(torch.long)
                X, l = append_feature(X, l, args.sustained_frames)

            if use_gpu:
                X, l, y = X.cuda(), l.cuda(), y.cuda()
                
            # compute CMVN
            X = (X - mean) * scale
            raw_y, y = y, F.one_hot(y, num_classes=num_classes).float()   
            if use_mixup:
                X, y, l, _ = mixup(X, y, l, mixup_alpha)
            if use_specaug:
                X = specaug_torch(X, speaug_param)
            
            pred = net(X) # bs x len x num_class
            
            pre_pad = F.pad(pred.permute(0, 2, 1), [7, 8])
            pred_avg = F.avg_pool1d(pre_pad, 16, 1)
            pred_avg = pred_avg.permute(0, 2, 1)
            
            ## Pooling
            # 做池化，得到num_class * 1大小的向量，可以和onehot做损失
            
            # 前5个epoch限制唤醒触发时间，后面放开限制
            if args.has_boundary and epoch < 15:
                # pred_t = maxpool_mask(pred, l, raw_y, ends, sustained_frames)
                # pred_t_avg = maxpool_mask(pred_avg, l, raw_y, ends, sustained_frames)
                pred_t = max_pooling_new(pred, l, raw_y, ends, sustained_frames)
                pred_t_avg = max_pooling_new(pred_avg, l, raw_y, ends, sustained_frames)
            else:
                pred_t = maxpool_mask(pred, l)
                pred_t_avg = maxpool_mask(pred_avg, l)

            # loss计算
            loss = loss_fn(pred_t, y)
            loss_avg = loss_fn(pred_t_avg, y)
            
            # 每帧损失和平均后的每16帧损失一起参与计算
            loss = (loss + loss_avg) * 0.5
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 10 == 0 and batch != 0:
                loss = loss.cpu().item()
                if not use_mixup:
                    acc = (pred_t.argmax(1) == raw_y).type(torch.float).sum().item() / len(X)
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} loss: {loss:>7f} acc: {(100*acc):>0.4f} [{epoch:>3d}/{epochs:>3d}]  [{batch:>5d}/{size:>5d}]")
                else:
                    acc = (pred_t.argmax(1) == raw_y).type(torch.float).sum().item() / len(X)
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} loss: {loss:>7f} acc: {(100*acc):>0.4f} [{epoch:>3d}/{epochs:>3d}]  [{batch:>5d}/{size:>5d}]")
                # break
        
        # test loop
        net = net.eval()
        size = 0
        num_batches = len(val_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch_data in val_dataloader:
                if not args.has_boundary:
                    X, l, y = batch_data
                else:
                    X, l, y, _, _ = batch_data
                X, l, y = X.squeeze(0), l.squeeze(0), y.squeeze(0)
                if args.append_feature:
                    l = l.type(torch.long)
                    X, l = append_feature(X, l, args.sustained_frames)
                size += X.shape[0]
                if use_gpu:
                    X, l, y = X.cuda(), l.cuda(), y.cuda()
                X = (X - mean) * scale
                pred = net(X) # bs x len x num_class
                
                pred_t = maxpool_mask(pred, l)
                batch_loss = loss_fn(pred_t, y).item()
                batch_correct = (pred_t.argmax(1) == y).type(torch.float).sum().item()
                test_loss += batch_loss
                correct += batch_correct

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.5f}%, Avg loss: {test_loss:>8f} \n")
            writer.add_scalar('epoch/cv_loss', test_loss, epoch)
            writer.add_scalar('epoch/cv_acc', correct, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
        scheduler.step(test_loss)
        save_pt(net, epoch, correct, test_loss, top_corrects, args.output_dir)
    
    writer.close()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data', help='training tfrecord list file',
                        required=True)
    parser.add_argument('--val_data', help='validation tfrecord list file')
    parser.add_argument('--cmvn', help='cmvn json file', required=True)
    parser.add_argument('--output_dir',
        help='directory to store training checkpoints and logs',
        required=True)
    parser.add_argument('--hotwords',
                        nargs='+',
                        help='hotword texts, space separated if more than one',
                        required=True)
    parser.add_argument('--use_gpu', action="store_true", default=False, help='training with GPU')
    parser.add_argument('--tcn_sep', action="store_true", default=False, help='use tcn sep model')
    parser.add_argument('--residual', action="store_true", default=False, help='model have residual connection')
    parser.add_argument('--gpu_ids', type=int, nargs='+', help='use GPU ids')
    parser.add_argument('--use_mixup', action="store_true", default=False, help='training with mixup')
    parser.add_argument('--use_specaug', action="store_true", default=False, help='training with specaug')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='begining learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='time of 1024')
    parser.add_argument('--pretrained_model', default=None, help='use pretrained model')
    parser.add_argument('--append_feature', action="store_true", default=False, help='append feature')
    parser.add_argument('--fuse_model', action="store_true", default=False, help='merge conv and batchnorm')
    parser.add_argument('--has_boundary', action="store_true", default=False, help='does label has boundary')
    parser.add_argument('--sustained_frames', type=int, default=50, help='number of frames sustained after wake up')
    args = parser.parse_args()
    
    args.hotwords = ['GARBAGE'] + args.hotwords
    
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train(args)
    
    
if __name__ == "__main__":
    main()