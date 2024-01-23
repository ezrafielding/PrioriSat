import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.modules.module import Module

class PositionalEncoder(nn.Module):
    # Abishek's code
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).to(x.device)
        return x

class GraphConvolutionLayer(Module):
    # Abishek's code
    def __init__(self, N, in_features, u, activation, dropout_rate=0.):
        # N: number of nodes, in_features: input features, u: number of hidden units in the graph conv
        # This layer is essentially one graph convolution step
        super(GraphConvolutionLayer, self).__init__()
        self.u = u
        self.in_features = in_features
        self.adj_list = nn.Linear(in_features, u)
        self.linear_2 = nn.Linear(in_features, u)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.pe = PositionalEncoder(in_features, N)

    def forward(self, adj_tensor, h_tensor=None):
        # The h_tensor is to support output from prev graph conv layers. If it is None, it is the first layer
        # So we just pass positional encodings
        if h_tensor is not None:
            annotations = h_tensor
        else:
            annotations = self.pe(torch.zeros((adj_tensor.shape[0], adj_tensor.shape[1], self.in_features), device=adj_tensor.device, dtype=adj_tensor.dtype))

        output = self.adj_list(annotations)
        output = torch.matmul(adj_tensor, output)
        out_linear_2 = self.linear_2(annotations)
        output = output + out_linear_2
        output = self.activation(output) if self.activation is not None else output
        output = self.dropout(output)
        return output

    
class MultiGraphConvolutionLayers(Module):
    # Abishek's code
    def __init__(self, N, in_features, conv_hid_dims, activation, dropout_rate=0.):
        # N: number of nodes, in_features: input features, conv_hid_dims: number of hidden units in the graph conv
        # Multiple graph convolution steps here, so conv_hid_dims is a list of hidden units
        super(MultiGraphConvolutionLayers, self).__init__()
        self.conv_nets = nn.ModuleList()
        self.units = conv_hid_dims
    
        for u0, u1 in zip([in_features] + self.units[:-1], self.units):
            self.conv_nets.append(GraphConvolutionLayer(N, u0, u1, activation, dropout_rate))

    def forward(self, adj_tensor, h_tensor=None):
        hidden_tensor = h_tensor
        for conv_idx in range(len(self.units)):
            hidden_tensor = self.conv_nets[conv_idx](adj_tensor, hidden_tensor)
        return hidden_tensor


class GraphConvolution(Module):
    # Abishek's code
    def __init__(self, N, in_features, graph_conv_units, dropout_rate=0.):
        # N: number of nodes, in_features: input features, graph_conv_units: number of hidden units in the graph conv
        # This is a weapper around the multi graph convolution layers (idk why this exists additionally, but fine)
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.graph_conv_units = graph_conv_units
        self.activation_f = torch.nn.Tanh()
        self.multi_graph_convolution_layers = \
            MultiGraphConvolutionLayers(N, in_features, self.graph_conv_units, self.activation_f, dropout_rate)

    def forward(self, adj_tensor, h_tensor=None):
        output = self.multi_graph_convolution_layers(adj_tensor, h_tensor)
        return output

class GraphAggregation(Module):
    # Abishek's code
    def __init__(self, N, in_features, aux_units, activation, dropout_rate=0.):
        # N: number of nodes, in_features: input features (input is from graph convolution: B x N x D), aux_units: number of output units in projection
        # Inputs are projected to aux_units and are summed together
        super(GraphAggregation, self).__init__()
        self.activation = activation
        self.i = nn.Sequential(nn.Linear(in_features, aux_units),
                                nn.Sigmoid())
        j_layers = [nn.Linear(in_features, aux_units)]
        if self.activation is not None:
            j_layers.append(self.activation)
        self.j = nn.Sequential(*j_layers)
        self.dropout = nn.Dropout(dropout_rate)
        # self.pe = PositionalEncoder(in_features, N)

    def forward(self, out_tensor, h_tensor=None):
        annotations = out_tensor
        if h_tensor is not None:
            annotations = torch.cat((out_tensor, h_tensor), -1)
        # The i here seems to be an attention.
        i = self.i(annotations)
        j = self.j(annotations)
        output = torch.sum(torch.mul(i, j), 1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dropout(output)

        return output

class MultiDenseLayer(Module):
    def __init__(self, aux_unit, linear_units, activation=None, dropout_rate=0.):
        # aux_unit: input features, linear_units: hidden dims (list becuase there can be multiple mlp layers)
        super(MultiDenseLayer, self).__init__()
        layers = []
        for c0, c1 in zip([aux_unit] + linear_units[:-1], linear_units):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.LayerNorm(c1))
            if activation is not None:
                layers.append(activation)
        self.linear_layer = nn.Sequential(*layers)

    def forward(self, inputs):
        h = self.linear_layer(inputs)
        return h