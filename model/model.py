import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from base import BaseModel

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class iDPath(BaseModel):
    def __init__(self, node_num, type_num, adj, emb_dim=16, gcn_layersize=[16, 16, 16], dropout=0.5):
        super().__init__()
        self.node_num = node_num
        self.adj = adj
        self.emb_dim = emb_dim
        self.value_embedding = nn.Embedding(node_num+1, emb_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(type_num+1, emb_dim, padding_idx=0)
        self.gcn = GCN(nfeat=gcn_layersize[0], nhid=gcn_layersize[1],
                       nclass=gcn_layersize[2], dropout=dropout)
        self.lstm = nn.LSTM(input_size=emb_dim*2, hidden_size=emb_dim)
        
        self.node_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.node_attention_softmax = nn.Softmax(dim=1)

        self.path_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.path_attention_softmax = nn.Softmax(dim=1)

        self.output_linear = nn.Linear(in_features=emb_dim, out_features=1)
        
    def forward(self, path_feature, type_feature, lengths, mask, gcn=True):
        # shape of path_feature: [batch_size, path_num, path_length]
        # shape of type_feature: [batch_size, path_num, path_length]
        '''GCN embedding'''
        total_node = torch.LongTensor([list(range(self.node_num+1))]).to(path_feature.device)
        ego_value_embedding = self.value_embedding(total_node).squeeze()
        if gcn:
            gcn_value_embedding = self.gcn(x=ego_value_embedding, adj=self.adj.to(path_feature.device))
        else:
            gcn_value_embedding = ego_value_embedding
        
        '''Embedding'''
        batch, path_num, path_len = path_feature.size()
        path_feature = path_feature.view(batch*path_num, path_len)
        # shape of path_embedding: [batch_size*path_num, path_length, emb_dim]
        path_embedding = gcn_value_embedding[path_feature]
        type_feature = type_feature.view(batch*path_num, path_len)
        # shape of type_embedding: [batch_size*path_num, path_length, emb_dim]
        type_embedding = self.type_embedding(type_feature).squeeze()
        # shape of feature: [batch_size*path_num, path_length, emb_dim]
        feature = torch.cat((path_embedding, type_embedding), 2)

        '''Pack padded sequence'''
        feature = torch.transpose(feature, dim0=0, dim1=1)
        feature = utils.rnn.pack_padded_sequence(feature, lengths=list(lengths.view(batch*path_num).data),
                                                 enforce_sorted=False)
        
        '''LSTM'''
        # shape of lstm_out: [path_length, batch_size*path_num, emb_dim]
        lstm_out, _ = self.lstm(feature)
        # unpack, shape of lstm_out: [batch_size*path_num, path_length, emb_dim]
        lstm_out, _ = utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=path_len)
        
        '''Node attention'''
        # shape of output_path_embedding: [batch_size*path_num, emb_dim]
        mask = mask.view(batch*path_num, path_len)
        output_path_embedding, node_weight_normalized = self.node_attention(lstm_out, mask)
        # the original shape of node_weight_normalized: [batch_size*path_num, path_length]
        node_weight_normalized = node_weight_normalized.view(batch, path_num, path_len)
        # shape of output_path_embedding: [batch_size, path_num, emb_dim]
        output_path_embedding = output_path_embedding.view(batch, path_num, self.emb_dim)
        
        '''Path attention'''
        # shape of output_path_embedding: [batch_size, emb_dim]
        # shape of path_weight_normalized: [batch_size, path_num]
        output_embedding, path_weight_normalized = self.path_attention(output_path_embedding)
                
        '''Prediction'''
        output = self.output_linear(output_embedding)
        
        return output, node_weight_normalized, path_weight_normalized

    def node_attention(self, input, mask):
        # the shape of input: [batch_size*path_num, path_length, emb_dim]
        weight = self.node_attention_linear(input) # shape: [batch_size*path_num, path_length, 1]
        # shape: [batch_size*path_num, path_length]
        weight = weight.squeeze() 
        '''mask'''
        # the shape of mask: [batch_size*path_num, path_length]
        weight = weight.masked_fill(mask==0, torch.tensor(-1e9))
        # shape: [batch_size*path_num, path_length]
        weight_normalized = self.node_attention_softmax(weight) 
        # shape: [batch_size*path_num, path_length, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2) 
        # shape: [batch_size*path_num, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1) 
        return input_weighted, weight_normalized

    def path_attention(self, input):
        # the shape of input: [batch_size, path_num, emb_dim]
        weight = self.path_attention_linear(input)
        # [batch_size, path_num]
        weight = weight.squeeze()
        # [batch_size, path_num]
        weight_normalized = self.path_attention_softmax(weight)
        # [batch_size, path_num, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2)
        # [batch_size, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1)
        return input_weighted, weight_normalized