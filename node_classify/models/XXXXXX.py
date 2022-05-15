import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in, n_h,numb,chu, activation,k_numb1,k_numb2,h1,h2,data_name,emd,k0):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h,numb,chu, activation,k_numb1,k_numb2,emd,k0)

        self.gcn2 = GCN(n_in, n_h,numb,chu, activation,k_numb1,k_numb2,emd,k0)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        
        self.h1 = h1
        self.h2 = h2
        self.data_name = data_name

    def forward(self, seq1, seq2, adj, sparse,training, msk, samp_bias1, samp_bias2):

        h_1_0 = self.gcn2(seq1, adj, sparse,training)
        h_2_0 = self.gcn2(seq2, adj, sparse,training)

        h_1 = self.gcn(seq1, adj, sparse,training)
        # if self.data_name != 'cora':
        if self.data_name != 'citeseer':
            h_1 = h_1 + self.h2*h_2_0
            # h_1 = h_1 + self.h1*h_1_0

        c = self.read(h_1, msk)
        c = self.sigm(c)


        h_2 = self.gcn(seq2, adj, sparse,training)

        if self.data_name == 'citeseer':
            h_2 = h_2 + self.h2*h_2_0 + self.h1*h_1_0  #0.2 0.5  最好结果
        else:
            # h_2 = h_2 + self.h1 * h_1_0 #+ self.h2 * h_2_0
            h_2 = h_2 + self.h1 * h_1_0

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj,diff, sparse,training, msk):
        h_1 = self.gcn(seq, adj, sparse,1)
        # h_2 = self.gcn(seq, diff, sparse,0)
        c = self.read(h_1, msk)

        return (h_1).detach(), c.detach()

