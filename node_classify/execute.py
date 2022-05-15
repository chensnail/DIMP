import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import os
from models.dimp import DIMP
from models.logreg import LogReg
from utils import process
import argparse
from utils.make__dataset import get_dataset
from utils.cluster import node_cluster
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime

parser = argparse.ArgumentParser()

# parser.add_argument('--dataset',type=str,default='pubmed',help='data name')
# parser.add_argument('--dataset',type=str,default='citeseer',help='data name')
parser.add_argument('--dataset',type=str,default='cora',help='data name')
parser.add_argument('--data_path',type=str,default='new_data/',help='data name')
# parser.add_argument('--dataset',type=str,default='citeseer',help='data name')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--nb_epochs',type=int,default=200,help='run all epochs')
parser.add_argument('--patience',type=int,default=20,help='when less loss ')
parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--hid_units',type=int,default=512)
parser.add_argument('--nonlinearity',type=str,default='prelu',help='special name to separate parameters')
parser.add_argument('--numb_start',type=int,default=2,help='EM循环开始')
parser.add_argument('--numb_end',type=int,default=3,help='EM循环结束')
parser.add_argument('--chu_start',type=float,default=1,help='EM参数1开始')
parser.add_argument('--chu_end',type=float,default=1.01,help='EM参数1结束')
parser.add_argument('--chu_strip',type=float,default=0.1,help='EM参数1结束')
parser.add_argument('--h1',type=float,default=-0.2,help='')
parser.add_argument('--h2',type=float,default=-0.2,help='')
parser.add_argument('--k_numb1',type=int,default=0,help='')
parser.add_argument('--k_numb2',type=float,default=1,help='')
parser.add_argument('--result_value',type=str,default='cora_value',help='保存结果值文件名')
parser.add_argument('--no_cuda',type=int,default=1,help='1 cuda不可用,0 cuda可用')
parser.add_argument('--standardize_graph',type=bool,default=True,help='1 cuda不可用,0 cuda可用')
parser.add_argument('--wd',type=float,default=0.0001,help='L2正则化')
parser.add_argument('--emd',type=float,default=0.8,help='')
parser.add_argument('--k0',type=float,default=0.1,help='')
parser.add_argument('--numb_epoch',type=int,default=10,help='')
parser.add_argument('--percent',type=float,default=0.1,help='')

args = parser.parse_args()

if args.dataset in ['cora','citeseer']:
    args.no_cuda = 0

l2_coef = 0.0
drop_prob = 0.0
sparse = True

def main(count,numb,chu):
    if args.dataset in ['cora', 'citeseer','pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test,graph = process.load_data(args.dataset)
        diff = process.compute_ppr(graph)
    else:
        adj, features, labels, idx_train, idx_val, idx_test = get_dataset(args.dataset, args.data_path+args.dataset+'.npz',args.standardize_graph, None)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if args.percent<1:
        idx_train, idx_val, idx_test = process.split_dataset(nb_nodes,labels,args.percent,True)

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        # features = process.sparse_mx_to_torch_sparse_tensor(features)
        # features = features.to_dense()
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    if args.dataset not in ['cora', 'citeseer', 'pubmed']:
        features = process.sparse_mx_to_torch_sparse_tensor(features)
        features = features.to_dense()
    else:
        features, _ = process.preprocess_features(features)

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])

    diff = torch.FloatTensor(diff[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    model = DIMP(ft_size, args.hid_units,numb, chu, args.nonlinearity,args.k_numb1,args.k_numb2,args.h1,args.h2,args.dataset,args.emd,args.k0)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_coef)

    if args.no_cuda == 0 and torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        if sparse:
            sp_adj = sp_adj.cuda()
        else:
            adj = adj.cuda()

        diff = diff.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    training = True
    for epoch in range(args.nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        diff = diff[:,idx,:]

        lbl_1 = torch.ones(args.batch_size, nb_nodes)
        lbl_2 = torch.zeros(args.batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if args.no_cuda == 0 and torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
        start_time = datetime.datetime.now()
        logits = model(features, shuf_fts, sp_adj if sparse else adj,sparse,training, None, None, None)

        loss = b_xent(logits, lbl)

        # print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))

    embeds, _ = model.embed(features, sp_adj if sparse else adj,diff.squeeze(0), sparse,False, None)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    train_embs = embeds[0, idx_train]
    # val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    # val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    # &&&&&&&&&&&&&&&
    #画图
    # colors = [
    #     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
    # ]
    #
    # z = TSNE(n_components=2).fit_transform(test_embs.cpu().numpy())
    # y = test_lbls.cpu().numpy()
    #
    # plt.figure(figsize=(8, 8))
    # for i in range(nb_classes):
    #     plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    # plt.axis('off')
    # plt.show()



    # plot_points(colors)


    #&&&&&&&&&&&&&



    # if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.percent>=1:
    #     accuracy, micro_f1, macro_f1, ARI, NMI = node_cluster(test_embs,test_lbls)

    tot = torch.zeros(1)
    if args.no_cuda == 0 and torch.cuda.is_available():
        tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(args.hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=args.wd)
        if args.no_cuda == 0 and torch.cuda.is_available():
            log.cuda()

        for _ in range(300):
            log.train()
            opt.zero_grad()

            logits = log(train_embs).squeeze(0)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs).squeeze(0)
        preds = torch.argmax(logits, dim=1)

        # accuracy, micro_f1, macro_f1, ARI, NMI = node_cluster(logits, test_lbls)

        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        # print(acc)
        tot += acc

    # print('Average accuracy:', tot / 50)

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.percent >= 1:
        return accs.mean().item(),accuracy, micro_f1, macro_f1, ARI, NMI
    else:
        return accs.mean().item()
    # return accs.mean().item()

if __name__=='__main__':

    accuracy_list = []
    micro_f1_list = []
    macro_f1_list = []
    ARI_list = []
    NMI_list = []
    cluster_list = []
    val_list = []
    torch.cuda.set_device(0)
    # args.nb_epochs = 20
    # args.patience = 50

    for numb in range(args.numb_start,args.numb_end):
        for chu in np.arange(args.chu_start,args.chu_end,args.chu_strip):
            # for hh1 in np.arange(-0.,0.9,0.1):
            #     args.h1 = hh1
            #     for hh2 in np.arange(0.5,0.9,0.1):
            #         args.h2 = hh2

            accuracy_list = []
            micro_f1_list = []
            macro_f1_list = []
            ARI_list = []
            NMI_list = []
            cluster_list = []
            val_list = []

            with open(args.result_value+'/data_'+str(numb)+'_'+str(chu)+'.txt', 'a+') as f:
            # with open(args.result_value+'/data_'+str(numb)+'_'+str(chu)+'_h1_'+str(hh1)+'_h2_'+str(hh2)+'.txt', 'a+') as f:
                for i in range(args.numb_epoch):
                    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.percent >= 1:
                        val,accuracy, micro_f1, macro_f1, ARI, NMI = main(i,numb,chu)
                        accuracy_list.append(float('%.3f' % accuracy))
                        micro_f1_list.append(float('%.3f' % micro_f1))
                        macro_f1_list.append(float('%.3f' % macro_f1))
                        ARI_list.append(float('%.3f' % ARI))
                        NMI_list.append(float('%.3f' % NMI))

                        cclu = ('Acc: {:.3f}%, Micro F1: {:.3f}%, Macro F1: {:.3f}%, ARI: {:.3f}%, NMI: {:.3f}%\n'.format(accuracy,micro_f1,macro_f1,ARI,NMI))
                        # cluster_list.append(cclu)
                        f.write(cclu)
                    else:
                        val = main(i, numb, chu)

                    val_list.append(float('%.2f' % val))

                    # f.write(cclu)
                    # f.write('\n')

                    # f.write(str(val))
                    # f.write('\n')
                    # if val<80:
                    #     break

                # f.write(str(cluster_list)+'\n')
                if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.percent >= 1:
                    f.write(str('accuracy:'+str(accuracy_list)+',accuracy_means:'+str('%.2f' % np.mean(accuracy_list))+',accuracy_max:'+str(max(accuracy_list)))+'\n')
                    f.write(str('micro_f1:'+str(micro_f1_list)+',micro_f1_means:'+str('%.2f' % np.mean(micro_f1_list))+',micro_f1_max:'+str(max(micro_f1_list)))+'\n')
                    f.write(str('macro_f1:'+str(macro_f1_list)+',macro_f1_means:'+str('%.2f' % np.mean(macro_f1_list))+',macro_f1_max:'+str(max(macro_f1_list)))+'\n')
                    f.write(str('ARI:'+str(ARI_list)+',ARI_means:'+str('%.2f' % np.mean(ARI_list))+',ARI_max:'+str(max(ARI_list)))+'\n')
                    f.write(str('NMI:'+str(NMI_list)+',NMI_means:'+str('%.2f' % np.mean(NMI_list))+',NMI_max:'+str(max(NMI_list)))+'\n')
                f.write(str('val:'+str(val_list)+',val_means:'+str('%.2f' % np.mean(val_list))+',val_max:'+str(max(val_list)))+'\n')
