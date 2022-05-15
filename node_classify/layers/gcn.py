import torch
import torch.nn as nn
import random
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft,numb,chu, act,k_numb1,k_numb2,emd,k0, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.fc2 = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.numb = numb
        self.chu = chu
        self.k_numb1 = k_numb1
        self.k_numb2 = k_numb2
        self.emd = emd
        self.kk = k0
        self.mm = nn.Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        self.mm.data.uniform_(-1,1)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False,flag=1):
        # nnn = seq.shape[0]
        # drop_rate = 0.2
        # drop_rates = torch.FloatTensor(np.ones(nnn) * drop_rate)
        # if training:
        #     masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)  # 修改这里代码
        #     seq = masks.cuda() * seq.cuda()
        # else:
        #     seq = seq * (1. - 0.2)

        k = self.fc(seq).squeeze(0)
        k2 = self.fc2(seq).squeeze(0)
        k += self.mm * k2
        # k = self.act(k)
        # k = torch.sigmoid(k)
        k = torch.relu(k)
        k0 = k

        if sparse:
            for n in range(1, self.numb):
                kappa = k.sum(0)
                # kappa = (kappa==0)*1+kappa

                # if flag == 1:
                k = self.emd*torch.spmm(adj, k)+self.k_numb1*k
                # else:
                #     k = self.emd*torch.mm(adj, k)+self.k_numb1*k

                D = torch.sum((k) / (kappa*self.chu), dim=1)
                # D = (D==0)*1+D
                rrr = torch.mm(D.unsqueeze(-1), kappa.unsqueeze(0))

                # for i in range(rrr.shape[0]):
                #     rrr[i] = (rrr[i]==0)*1+rrr[i]
                # rrr = rrr + torch.eye(rrr.shape[0],rrr.shape[1]).cuda()*(1)

                k = (k/ rrr)+(self.k_numb2*k) +self.kk*k0
            k = torch.unsqueeze(k,0)
        else:
            k = torch.bmm(adj, k)
        if self.bias is not None:
            k += self.bias


        return self.act(k)
