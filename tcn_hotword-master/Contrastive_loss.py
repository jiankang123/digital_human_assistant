import torch
import torch.nn.functional as F
import numpy

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, batch_embedding, labels):
        norm_batch = F.normalize(batch_embedding,dim=1)
        # batch size
        n = labels.shape[0]

        sim_matrix = F.cosine_similarity(norm_batch.unsqueeze(1),norm_batch.unsqueeze(0),dim=2)
        # pos mask 相同label取1，不同label取0
        mask = torch.ones_like(sim_matrix)*(labels.expand(n,n).eq(labels.expand(n,n).t())).cuda()
        # print('mask:\n{}'.format(mask))

        # neg mask = 1 - pos_mask
        neg_mask = torch.ones_like(mask)-mask
        
        diag_mask = 1-torch.eye(n,n)
        diag_mask = diag_mask.cuda()

        exp_sim_matrix = torch.exp(sim_matrix/self.temperature)

        # 不计算对角位置的损失（自己和自己不产生对比）
        exp_sim_matrix = exp_sim_matrix * diag_mask
        # print('exp_sim_matrix:\n{}'.format(exp_sim_matrix))


        pos_matrix = mask * exp_sim_matrix
        neg_matrix = exp_sim_matrix - pos_matrix
        # print('pos_matrix:\n{}'.format(pos_matrix))

        # 计算对比损失分母，分母由两部分组成，一个是样例正对，一个是样例负对和
        # 每行（也就是每个样例）的负对和，扩展为相似度矩阵大小
        neg_sim_sum = torch.sum(neg_matrix, dim=1)
        neg_sim_sum_expand = neg_sim_sum.repeat(n,1).T
        # print('neg_sim_sum_expand:\n{}'.format(neg_sim_sum_expand))

        # 正对加负对和得到对比损失分母
        sim_sum = pos_matrix + neg_sim_sum_expand
        # print('sim_sum:\n{}'.format(sim_sum))

        # pos_matrix中只有正对有值，其他位置为0
        # 除的结果只有正对位置是pos/(pos+neg)，其他位置为0
        loss = torch.div(pos_matrix, sim_sum)
        # 0取log会出错，所以在0的位置上+1，log(1)=0，后续求和可以去除这些位置
        loss = neg_mask + loss + torch.eye(n,n).cuda()
        # print('loss:\n{}'.format(loss))

        loss = -torch.log(loss)
        # print('loss:\n{}'.format(loss))
        # 每个样例损失求和/样例数，得到最终损失
        loss = torch.sum(torch.sum(loss, dim=1))/(len(torch.nonzero(loss)))
        # print('loss:\n{}'.format(loss))
        return loss
        

if __name__ == '__main__':
    lossfunc = ContrastiveLoss()
    batch_data = torch.rand(1024,64).cuda()
    labels = torch.randint(0,6,(1024,)).cuda()
    # print(batch_data.shape)
    # print(labels.shape)
    loss = lossfunc(batch_data,labels)


        
