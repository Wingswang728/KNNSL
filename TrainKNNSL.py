# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:38:49 2022

@author:wj
"""

import sys
import pandas as pd
import networkx as nx
import numpy as np
from SL_nn import *
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score,f1_score,recall_score,precision_score
import torch
from numpy import *

file_gene=pd.read_csv('978特征矩阵-A549.csv')
geneid=file_gene.iloc[:,0].tolist()
geneid=[str(gene) for gene in geneid]

# print(len(geneid),geneid)
null=0
# find where object str appears in a char
def getstrindex(stri,obj):
    i=0
    index=[]
    while True:
        i=stri.find(str(obj),i)

        if i == -1:
            break
        index.append(i)
        i+=1
    return index

def findgo(stri,index):
    gene_togo=[stri[i:i+10] for i in index]
    return gene_togo



# Evaluation metrics
def pr(pre, test, pre_p):
    pre = np.array(pre, dtype=np.float32)
    test = np.array(test, dtype=np.float32)
    pre_p = np.array(pre_p, dtype=np.float32)

    AUC = roc_auc_score(test, pre_p)
    AUPR = average_precision_score(test, pre_p)
    ACC1 = accuracy_score(test, pre)

    F1 = f1_score(test, pre)

    Pre = precision_score(test, pre)

    Recall = recall_score(test, pre)
    

    return AUC, AUPR, Pre, Recall, F1, ACC1
#When a node is deleted, connect all its parent nodes and child nodes.
def delnodes(term,dG):
    # Ancestor nodes
    ancestor = list(dG.predecessors(term))
    # parent nodes, close to root nodes
    parent = [edge[0] for edge in dG.in_edges(term)] 
    # child nodes,far from root nopdes
    child = [edge[1] for edge in dG.out_edges(term)] 
    if term in notin_gene:
        dG.remove_node(term)
    if len(parent)==0:
        print('thats impossible for',term)
    if term not in geneid:
        for par in parent:
            for chi in child:
                dG.add_edge(par,chi)
        dG.remove_node(term)
term_direct_gene={}
term_size_m={}
dG=nx.read_gml('wj-dG55.gml')

 # add the gene to direct_term,gene_term_map is what gene direct link to the term,count is the number of gene direct link to term
count={} 
gene_term_map={} 

notin_term=[] # error term,do not link to the root term
for term in dG.nodes():

    if term not in geneid:
        count[term]=0
        deslist=nxadag.descendants(dG,term)
        anclist=nxadag.ancestors(dG,term)
        gene_term_map[term]=[]
        if 'GO:0008150' not in anclist:
            notin_term.append(term)
            print('not in',term)

        for child in deslist:
            if child in geneid:

                count[term]+=1
                gene_term_map[term].append(child)
    else:
        continue
print('count',count['GO:0008150'])
print('len not in',len(notin_term))


for term in dG.nodes():
    anclist = nxadag.ancestors(dG, term)
    child = [edge[1] for edge in dG.out_edges(term)] 
    if 'GO:0008150' not in anclist:
        print(term,'after not in ')

    direct_gene=[chi for chi in child if chi in geneid]
    if term not in geneid:
        term_size_m[term] = int(count[term])
    if len(direct_gene)!=0:
        term_direct_gene[term]=direct_gene
print(len(term_direct_gene),len(term_direct_gene['GO:0008150']))

root='GO:0008150'
term_direct_gene_map=term_direct_gene
term_size_map=term_size_m

gene2ind=pd.read_csv('gene2id.csv')   
gene2ind["ID"] = gene2ind["ID"].astype("int")
gene2ind["gene"] = gene2ind["gene"].astype("int")
gene2ind=gene2ind.values

gene2ind_map={}
for index,gene in gene2ind:
    gene2ind_map[str(gene)]=index
for term in term_direct_gene_map:
    for index,gene in enumerate(list(term_direct_gene_map[term])):
        term_direct_gene_map[term][index]=gene2ind_map[gene] # Convert to 0-797ID

# remove gene ndoes
for term in list(dG.nodes()):
    if term in geneid:
        dG.remove_node(term)

wj_leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0] #底层非基因
i=0


print(len(wj_leaves),wj_leaves)


print('this is the second part')

from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold



cuda = 0
CUDA_ID = cuda

train_epoch=400
batch_size=64
lr=1e-4


gene_dim=1956
num_hiddens_genotype=2

# Initialize parameters 
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Mask out two disconnected edges 
def create_term_mask(term_direct_gene_map,gene_dim):
    term_mask_map={}
    for term,gene_set in term_direct_gene_map.items():
        mask=torch.zeros(len(gene_set),gene_dim)
        for i,gene_id in enumerate(gene_set):
            mask[:,gene_id]=1
            mask[:,gene_id+978]=1
        mask_gpu=torch.autograd.Variable(mask.cuda(CUDA_ID))
        term_mask_map[term]=mask_gpu
    return term_mask_map

#Features
def concat_feaAB_all(geneA_list,geneB_list,fea_all):
    feaAB_all = pd.DataFrame(index = list(fea_all.columns)+list(fea_all.columns))
    for i in range(len(geneA_list)):
        geneA = geneA_list[i]
        geneB = geneB_list[i]
        featureA = fea_all.loc[geneA]
        featureB = fea_all.loc[geneB]
        features_ = pd.concat([featureA,featureB],axis = 0)
        feaAB_all = pd.concat([feaAB_all,features_],axis = 1)
    indexlist = []
    for i in range(len(geneA_list)):
        indexlist.append(i)
    feaAB_all.columns=indexlist
    return feaAB_all.T

#Samples
samples = pd.read_csv('samples-AB-A549.csv')
geneA_list = samples['gene A']
geneB_list = samples['gene B']
labels = samples['label']
features = pd.read_csv('978特征矩阵-A549.csv',index_col=0)
features1 = features.T
featuresAB_1 = concat_feaAB_all(geneA_list,geneB_list,features1)
featuresAB_1=np.array(featuresAB_1)

data  = torch.tensor(featuresAB_1).float()
# finger len =1764
label=torch.tensor(labels).unsqueeze(1)

test_acc=0
skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=66) 
def train_model(root, term_size_map, term_direct_gene_map, dG,  gene_dim, 
                train_epochs,  learning_rate, num_hiddens_genotype,
                train_loader,test_loader,test_x,test_y,loop):
    epoch_start_time = time.time()  
    best_model = 0
    max_corr = 0

    model = SL_nn(term_size_map, term_direct_gene_map, dG, gene_dim, root, num_hiddens_genotype)
    model.apply(weight_init)

    AUC = []
    AUPR = []
    ACC1 = []
    Pre = []
    Recall = []
    F1 = []

    model.cuda(CUDA_ID)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

    optimizer.zero_grad()

    for name, param in model.named_parameters():  
        term_name = name.split('_')[0] 

        if '_direct_gene_layer.weight' in name:
            
            param.data = torch.mul(param.data, term_mask_map[term_name])*0.1
        else:  
            param.data = param.data*0.1 
    train_loader=train_loader

    test_loader=test_loader
    test_acc1=0
    for epoch in range(train_epochs):

        # Train
        
        total_acc=0
        model.train()
        train_predict = torch.zeros(0, 0).cuda(CUDA_ID)
        
        for i, (inputdata, labels) in enumerate(train_loader):
            
            cuda_features = torch.autograd.Variable(inputdata.cuda(CUDA_ID))
            cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID)).squeeze().long()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer

            # Here term_NN_out_map is a dictionary

            aux_out_map, _ = model(cuda_features)  # Forward propagation，acqure aux_out_map, term_NN_out_map

            train_predict = aux_out_map['GO:0008150'].data
            total_loss = 0
            for name, output in aux_out_map.items():  
                
                loss = nn.CrossEntropyLoss()
                if name == 'GO:0008150':
                    total_loss += loss(output, cuda_labels)
                else:  
                    total_loss += 0.2 * loss(output, cuda_labels)

            total_loss.backward()

            for name, param in model.named_parameters():  
                if '_direct_gene_layer.weight' not in name:  
                    continue
                term_name = name.split('_')[0]

                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            optimizer.step()

            train_predict_soft=torch.softmax(train_predict, dim=-1)
            train_predict_soft=torch.argmax(train_predict_soft,dim=-1)
            num_correct = (cuda_labels == train_predict_soft).sum().item()
            train_acc = num_correct / len(train_predict_soft)

            total_acc+=train_acc

        print(epoch,'this acc is ',total_acc/len(train_loader))

        #Prediction
        model.eval()

        test_predict = torch.zeros(0, 0).cuda(CUDA_ID)
        total_label=[]
        total_pre=[]
        test_total_acc=0

        test_x2 = torch.cat((test_x[:, 978:1956], test_x[:, 0:978]), 1)#To achieve same prediction for gene pairs in different order
        cuda_features1 = test_x.cuda(CUDA_ID)
        cuda_features2 = test_x2.cuda(CUDA_ID)
        cuda_labels = test_y.cuda(CUDA_ID).squeeze().long()
        aux_out_map1, _ = model(cuda_features1)
        aux_out_map2, _ = model(cuda_features2)
        test_predict1 = aux_out_map1['GO:0008150'].data
        test_predict2 = aux_out_map2['GO:0008150'].data
        test_predict = 0.5 * (test_predict1 + test_predict2)
        test_predict_soft1 = torch.softmax(test_predict, dim=-1)
        indices = torch.tensor([1]).cuda(CUDA_ID)
        test_predict_soft2 = torch.index_select(test_predict_soft1, -1, indices)
        test_predict_soft = torch.argmax(test_predict_soft1, dim=-1)
        num_correct_test = (cuda_labels == test_predict_soft).sum().item()
        test_acc = num_correct_test / len(test_predict_soft)
       
        pre = np.array(list(test_predict_soft), dtype=np.float32)     
        test = np.array(list(cuda_labels), dtype=np.float32)
        pre_p = np.array(list(test_predict_soft2), dtype=np.float32)
    
        AUC=roc_auc_score(test, pre_p)
        AUPR=average_precision_score(test, pre_p)
        ACC1=accuracy_score(test, pre)
        F1=f1_score(test, pre)
        Pre=precision_score(test, pre)    
        Recall=recall_score(test, pre)
        metric=[AUC,AUPR,ACC1,F1,Pre,Recall]
        print('this test acc is ', test_acc, 'pr is',
              [pr(list(test_predict_soft), list(cuda_labels), list(test_predict_soft2))])
        dic_test_acc.append(metric)

loop=0


for train_index,test_index in skf.split(data,label):
    loop+=1

    dic_test_acc=[]

    train_x, train_y = data[train_index], label[train_index]
    train_x = torch.cat((train_x, torch.cat((train_x[:, 978:1956], train_x[:, 0:978]), 1)), 0)#To achieve same predictions for gene pairs in deffrent order
    train_y = torch.cat((train_y, train_y), 0)
    test_x, test_y = data[test_index], label[test_index]
    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False
    )
    dG=nx.read_gml('wj-dG55.gml')
    for term in list(dG.nodes()):
        if term in geneid:
            dG.remove_node(term)
    train_model(root=root,term_size_map=term_size_map,term_direct_gene_map=term_direct_gene_map,dG=dG,gene_dim=gene_dim,train_epochs=train_epoch,
            learning_rate=lr,num_hiddens_genotype=num_hiddens_genotype,
            train_loader=train_loader,test_loader=test_loader,test_x=test_x,test_y=test_y,loop=loop)
    acc=[]
    for i in range(0,train_epoch):
        acc.append(dic_test_acc[i][2])
    print("----------------------------------------")
    print(loop)
    print("----------------------------------------")
    print(acc[train_epoch-1])

    print('ok')

