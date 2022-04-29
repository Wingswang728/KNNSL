import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SL_nn(nn.Module):

    def __init__(self, term_size_map, term_direct_gene_map, dG, ngene, root, num_hiddens_genotype):

        super(SL_nn, self).__init__()  
        self.root = root  
        self.num_hiddens_genotype = num_hiddens_genotype  
        self.term_direct_gene_map = term_direct_gene_map
        self.cal_term_dim(term_size_map)
        self.gene_dim = ngene 
        self.contruct_direct_gene_layer()
        self.term_layer_list = self.construct_NN_graph(dG)     
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {} 

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype 

            num_output = int(num_output)
            self.term_dim_map[term] = num_output
    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):
        for term, gene_set in self.term_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no directed asscoiated genes for', term)
                sys.exit(1)        
            self.add_module(term + '_direct_gene_layer',
                            nn.Linear(self.gene_dim, len(gene_set))) 
    def construct_NN_graph(self, dG):

        self.term_layer_list = []  
        self.term_neighbor_map = {}
        for term in dG.nodes(): 
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term): 
                self.term_neighbor_map[term].append(child)
        wzm_i = 0
        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]              
            wzm_i+=1
            print(wzm_i,'len',len(leaves))
            if len(leaves) == 0: 
                break
            self.term_layer_list.append(leaves) 
            for term in leaves:
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])
                term_hidden = self.term_dim_map[term]
                self.add_module(term + '_linear_layer', nn.Linear(input_size, term_hidden))
                self.add_module(term + '_batchnorm_layer', nn.BatchNorm1d(term_hidden),)
                self.add_module(term + '_aux_linear_layer1', nn.Linear(term_hidden, 2))
                self.add_module(term + '_aux_linear_layer2', nn.Linear(2, 2))
            dG.remove_nodes_from(leaves)  
        return self.term_layer_list
    # definition of forward function
    def forward(self, x):  # x is the input of the model

        gene_input = self.gene_dim

        term_gene_out_map = {}  

        for term, _ in self.term_direct_gene_map.items():
            term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](
                x)  

        term_NN_out_map = {}  
        aux_out_map = {}  
        for i, layer in enumerate(self.term_layer_list):
            for term in layer:
                # print(term)

                child_input_list = []

                for child in self.term_neighbor_map[term]:  
                    child_input_list.append(term_NN_out_map[child])

                if term in self.term_direct_gene_map:  
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1) 
                term_NN_out = self._modules[term + '_linear_layer'](child_input)

                Tanh_out = torch.tanh(term_NN_out)
                term_NN_out_map[term] = self._modules[term + '_batchnorm_layer'](Tanh_out)
                aux_layer1_out = torch.tanh(self._modules[term + '_aux_linear_layer1'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term + '_aux_linear_layer2'](aux_layer1_out)
        return aux_out_map, term_NN_out_map
