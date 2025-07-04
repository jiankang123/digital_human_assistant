import torch
import numpy as np

def max_pooling(frame_p, seq_len):
    # bs x num_class
    num_classes = frame_p.shape[-1]
    mask_list = []
    for i in seq_len:
        # print(i.item())
        mask = torch.ones((i.type(torch.int).item()))
        mask_list.append(mask)
    # pad到相同大小
    mask = torch.nn.utils.rnn.pad_sequence(
        mask_list, batch_first=True)
    # print('mask:{}'.format(mask.shape))
    mask = mask.unsqueeze(-1)

    # 把mask做到和pred相同size
    mask = torch.tile(mask, [1, 1, num_classes]).bool().to(frame_p.device)
    
    # inf是为了和后面最大池化损失搭配
    # 0类别的所有时间步在inf取最小，其他类别的所有时间步在-inf取最大
    score_mask_value = np.inf
    score_mask_value = score_mask_value * torch.ones_like(frame_p)
    score_mask_value = score_mask_value * torch.tensor([1] + [-1] * (num_classes - 1)).to(frame_p.device)
    # print('mask:{}'.format(mask.shape))
    # print('frame_p:{}'.format(frame_p.shape))
    # print('score_mask_value:{}'.format(score_mask_value.shape))
    real_frame_p = torch.where(mask, frame_p, score_mask_value)
    pred_l = [torch.min(real_frame_p[:, :, 0], 1).values]
    for i in range(1, num_classes):
        pred_l.append(torch.max(real_frame_p[:,:,i], 1).values)
    wav_p = torch.stack(pred_l).T
    return wav_p

def max_pooling_with_boundary(pred, seq_len, end):
    num_classes = pred.shape[-1]
    pred_t = torch.zeros((pred.shape[0], pred.shape[-1]), device=pred.device)
    pred_t[:, 0] = torch.min(pred[:, :, 0], 1).values
    for i in range(1, num_classes):
        for j in range(pred.shape[0]):
            if end[j] > 0:
                if end[j] < seq_len[j]:
                    pred_t[j][i] = pred[j, end[j]:seq_len[j], i].max()
                else:
                    pred_t[j][i] = pred[j, end[j]:, i].max()
            else:
                pred_t[j][i] = pred[j, :seq_len[j], i].max()
    return pred_t

def padding_mask(lengths, max_len=None):
    """
    Examples:
        >>> lengths = torch.tensor([2, 2, 3], dtype=torch.int32)
        >>> mask = padding_mask(lengths)
        >>> print(mask)
        tensor([[False, False,  True],
                [False, False,  True],
                [False, False, False]])
    """
    batch_size = lengths.size(0)
    if not max_len:
        max_len = int(lengths.max().item())
    seq = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq = seq.expand(batch_size, max_len)
    return seq >= lengths.unsqueeze(1)


def max_pooling_youhua(pred, seq_len, end):
    num_classes = pred.shape[-1]
    pred_t = torch.zeros((pred.shape[0], pred.shape[-1]), device=pred.device)
    
    mask = torch.zeros(pred.shape, device=pred.device)
    for i in range(pred.shape[0]):
        if end[i] > 0:
            mask[i, end[i]:seq_len[i], :] = 1
        else:
            mask[i, :seq_len[i], :] = 1


    mask = mask.bool()
    # score_mask_value = torch.inf
    # score_mask_value = score_mask_value * torch.ones_like(pred)
    score_mask_value = torch.full(pred.shape, torch.inf, device=pred.device)
    score_mask_value = score_mask_value * torch.tensor([1] + [-1] * (num_classes - 1)).to(pred.device)

    real_pred = torch.where(mask, pred, score_mask_value)
    pred_t[:, 0] = torch.min(real_pred[:, :, 0], 1).values
    pred_t[:,1:], _ = real_pred[:,:,1:].max(1)

    return pred_t


# 推荐使用这个函数
def maxpool_mask(pred, seq_len, label=None, end=None, sustained_frames=None):
    num_classes = pred.shape[-1]
    time_len = pred.shape[1]

    mask = torch.logical_not(padding_mask(seq_len, max_len=time_len))

    if end:
        end = torch.tensor(end, dtype=torch.int32).to(pred.device)
        # 往前移25帧
        end -= 25
        end_mask = padding_mask(end, max_len=time_len).to(pred.device)
        if sustained_frames:
            sustained_mask = torch.logical_not(padding_mask(
                end+sustained_frames, max_len=time_len)).to(pred.device)
            end_mask = torch.logical_and(end_mask, sustained_mask)

        # 若end大于len，可能是对齐的问题。对应的样本要放开边界。
        # 负样本边界也要放开
        if label is not None:
            row_mask = label == 0
        row_mask2 = end == 0
        row_mask = torch.logical_or(row_mask, row_mask2)
        row_mask = torch.logical_or(row_mask, seq_len < end)
        row_mask = torch.tile(row_mask.unsqueeze(-1), [1, time_len])
        end_mask = torch.logical_or(end_mask, row_mask)
        mask = torch.logical_and(mask, end_mask)

    mask = torch.tile(mask.unsqueeze(-1), [1, 1, num_classes]).to(pred.device)
    score_mask_value = torch.full(pred.shape, torch.inf, device=pred.device)
    score_mask_value *= torch.tensor([1] + [-1] * (num_classes - 1)).to(pred.device)

    real_pred = torch.where(mask, pred, score_mask_value)

    pred_l = [torch.min(real_pred[:, :, 0], 1).values]
    for i in range(1, num_classes):
        pred_l.append(torch.max(real_pred[:,:,i], 1).values)
    pred_t = torch.stack(pred_l).T
    return pred_t


# 新增逻辑（目标类做maxpool时取有效时间段最小值，迫使唤醒触发时间段加长），待优化执行速度
def max_pooling_new(pred, seq_len, label=None, end=None, sustained_frames=None):
    num_classes = pred.shape[-1]
    time_len = pred.shape[1]

    mask = torch.logical_not(padding_mask(seq_len, max_len=time_len))
    mask = torch.tile(mask.unsqueeze(-1), [1, 1, num_classes]).to(pred.device)
    score_mask_value = torch.full(pred.shape, torch.inf, device=pred.device)
    score_mask_value = score_mask_value * torch.tensor([1] + [-1] * (num_classes - 1)).to(pred.device)
    real_pred = torch.where(mask, pred, score_mask_value)

    pred_t = torch.zeros((pred.shape[0], pred.shape[-1]), device=pred.device)
    seq_len_list = seq_len.type(torch.int32).tolist()
    pred_t[:, 0] = torch.min(real_pred[:, :, 0], 1).values
    for i in range(1, num_classes):
        pred_t[:, i] = torch.max(real_pred[:, :, i], 1).values

    for i in range(1, num_classes):
        for j in range(pred.shape[0]):
            if end and end[j] > 0 and end[j] < seq_len_list[j]:
                if label[j] == i:
                    if seq_len_list[j] < end[j] + sustained_frames:
                        pred_t[j][i] = pred[j, end[j]:seq_len_list[j], i].min()
                    else:
                        pred_t[j][i] = pred[j, end[j]:end[j] + sustained_frames, i].min()
    return pred_t


def linear_softmax_pooling(frame_p, seq_len):
    # bs x num_class
    num_classes = frame_p.shape[-1]
    mask_list = []
    for i in seq_len:
        mask = torch.ones((i.item()))
        mask_list.append(mask)
    # pad到相同大小
    mask = torch.nn.utils.rnn.pad_sequence(
        mask_list, batch_first=True)
    mask = mask.unsqueeze(-1)
    # 把mask做到和pred相同size
    mask = torch.tile(mask, [1, 1, num_classes]).to(frame_p.device)
    
    # 将padding出来的值全部乘0
    # 在pooling时就不会影响真实值
    real_frame_p = frame_p * mask
    wav_p = torch.pow(real_frame_p, 2).sum(1) / real_frame_p.sum(1)
    return wav_p





if __name__ == "__main__":
    fp = torch.rand((2, 10, 5), dtype=torch.float32)
    l = torch.tensor([5,7], dtype=torch.int32)
    lsp1 = maxpool_mask(fp, l, end=None)
    lsp2 = maxpool_mask(fp, l, end=[1,2], sustained_frames=2)
    lsp3 = maxpool_mask(fp, l, end=[8,8], sustained_frames=2)
    print(fp)
    print(lsp1)
    print(lsp2)
    print(lsp3)