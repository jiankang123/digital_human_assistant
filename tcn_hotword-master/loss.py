import torch
import torch.nn.functional as F

def padding_mask(lengths):
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
    max_len = int(lengths.max().item())
    seq = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq = seq.expand(batch_size, max_len)
    return seq >= lengths.unsqueeze(1)


def max_pooling_loss_with_location(logits, target, lengths, end, 
                                   sustained_frame):
    mask = padding_mask(torch.tensor(lengths, dtype=torch.int32))
    num_utts = logits.size(0)
    num_keywords = logits.size(2)

    num_pos = 0
    target = target.cpu()
    loss = 0.0
    for i in range(num_utts):
        # negative sample
        prob = logits[i]
        if target[i] == 0:
            prob[:, 0] = prob[:, 0].masked_fill(mask[i], 1.0)
            prob[:, 1:] = 1 - prob[:, 1:].masked_fill(mask[i], 0.0)
            prob = torch.clamp(prob, 1e-8, 1.0)
            for j in range(num_keywords):
                loss += -torch.log(prob[:, j].min())
        # positive sample
        else:
            num_pos += 1
            # triggerred location
            m = mask[i].clone().detach()
            m[:end[i]] = True
            m[end[i]:end[i]+sustained_frame] = False
            m[end[i]+sustained_frame:] = True
            prob_pos = prob.clone()
            for j in range(num_keywords):
                if target[i] == j:
                    prob_pos[:, j] = prob_pos[:, j].masked_fill(m, 1.0)
                    prob_pos[:, j] = torch.clamp(prob_pos[:, j], 1e-8, 1.0)
                    loss += -torch.log(prob_pos[:, j].min())
                else:
                    prob_pos[:, j] = 1 - prob_pos[:, j].masked_fill(m, 0.0)
                    prob_pos[:, j] = torch.clamp(prob_pos[:, j], 1e-8, 1.0)
                    loss += -torch.log(prob_pos[:, j].min())
            
            # non-triggerred location treat as negative
            prob_neg = prob.clone()
            m = torch.logical_not(m)
            prob_neg[:, 0] = prob_neg[:, 0].masked_fill(m, 1.0)
            prob_neg[:, 1:] = 1 - prob_neg[:, 1:].masked_fill(m, 0.0)
            prob_neg = torch.clamp(prob_neg, 1e-8, 1.0)
            for j in range(num_keywords):
                loss += -torch.log(prob_neg[:, j].min())
            
    loss = loss / (num_utts + num_pos)
    return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        # 注意，这里的alpha是给定的一个list(tensor),
        # 里面的元素分别是每一个类的权重因子
        self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num

    def forward(self, predict, target):
        # softmmax获取预测概率
        pt = F.softmax(predict, dim=1) 
        if isinstance(target, int):
            target = torch.tensor(target)
        # 获取target的one hot编码
        class_mask = F.one_hot(target, self.class_num) 
        alpha = self.alpha[target]
        # 利用onehot作为mask，提取对应的pt
        probs = (pt * class_mask).sum(1).view(-1, 1) 
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        loss = loss.mean()
        return loss


class my_MCE(torch.nn.Module):
    def __init__(self, ):
        super(my_MCE, self).__init__()

    def forward(self, logits, labels):
        loss = -torch.mean(
                labels * torch.log(logits) +
                (1 - labels) * torch.log(1 - logits)
            )
        return loss


def triplet_loss(logits, labels):
    """
    Triplet Loss的损失函数
    """
    margin = 1.0
    def single_triplet_loss(anc, pos, neg):
        # 欧式距离
        pos_dist = torch.sum(torch.square(anc - pos), dim=-1, keepdims=True)
        neg_dist = torch.sum(torch.square(anc - neg), dim=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + margin
        loss = torch.max(basic_loss, torch.tensor([0.0], 
                         device=basic_loss.device))
        return loss
    loss_fun = torch.nn.TripletMarginLoss()

    predict = logits.argmax(1)
    labels = labels.argmax(1)

    loss = 0
    size = labels.shape[0]
    for i in range(size):
        if labels[i] != 0:
            for j in range(i+1, size):
                if labels[j] == labels[i]:
                    for k in range(size):
                        if (labels[k] != 0 and labels[k] != labels[j] 
                            and predict[k] == labels[j]):
                            loss += single_triplet_loss(logits[i], logits[j], 
                                                        logits[k])
    return loss


def test_ce():
    m = torch.nn.Sigmoid()
    loss = torch.nn.BCELoss()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(m(input), target)
    print(output)
    my_loss = my_MCE()
    my_output = my_loss(m(input), target)
    print(my_output)

    s = torch.nn.Softmax()
    print(m(input))
    print(s(input))



def test_FocalLoss():
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
    fl = FocalLoss(5, 2, alpha)
    pred = torch.rand((32, 5))
    target = 2
    loss = fl(pred, target)

if __name__ == "__main__":
    test_ce()