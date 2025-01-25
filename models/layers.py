import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.ODE import ODENet
from models.transformer import MultiHeadAttention, PositionalWiseFeedForward, PositionalEncoding
from models.utils import seq_mask, masked_softmax
from models.transformer import *
from gensim.models import FastText


class MyLinear(nn.Module):
    def __init__(self, output_dim):
        super(MyLinear, self).__init__()
        self.output_dim = output_dim
        self.linear = None

    def forward(self, x):
        if self.linear is None:
            self.linear = nn.Linear(x.size(-1), self.output_dim, device=x.device)
        output = self.linear(x)
        return output




class HierarchicalEmbedding(nn.Module):
    def __init__(self, code_levels, code_num_in_levels, code_dims, code_num):
        super(HierarchicalEmbedding, self).__init__()
        self.level_num = len(code_num_in_levels)
        self.code_levels = code_levels  # (leaf code num * level_num)
        self.level_embeddings = nn.ModuleList([nn.Embedding(code_num, code_dim)
                                               for code_num, code_dim in zip(code_num_in_levels, code_dims)])
        for emb in self.level_embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self):
        embeddings = [
            self.level_embeddings[level](self.code_levels[:, level])
            for level in range(self.level_num)
        ]
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings




class DrugEmbedding(nn.Module):
    def __init__(self, drug_num, drug_dim):
        super().__init__()
        self.drug_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(drug_num, drug_dim)))

    def forward(self):
        return self.drug_embeddings


class GraphConvBlock(nn.Module):
    def __init__(self, node_type, input_dim, output_dim, adj):
        super().__init__()
        self.node_type = node_type
        self.adj = adj
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()

        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, embedding, embedding_neighbor, code_adj=None):
        output = embedding + torch.matmul(self.adj, embedding_neighbor)
        if self.node_type == 'code':

            assert code_adj is not None
            output += torch.matmul(code_adj, embedding)

        output = self.dense(output)
        output = self.bn(output)
        output = self.activation(output)

        return output


def norm_no_nan(x):
    sum_x = torch.sum(x, dim=-1, keepdim=True)
    return torch.nan_to_num(torch.div(x, sum_x), 0)


class GraphConvolution(nn.Module):
    def __init__(self, drug_dim, code_dim,
                 drug_code_adj, code_code_adj,
                 drug_hidden_dims, code_hidden_dims, device):
        super().__init__()
        self.drug_code_adj = norm_no_nan(drug_code_adj)
        self.code_drug_adj = norm_no_nan(drug_code_adj.transpose(-1, -2))
        self.code_code_adj = code_code_adj
        self.drug_blocks = []
        self.code_blocks = []
        last = drug_dim
        for layer, dim in enumerate(drug_hidden_dims):
            self.drug_blocks.append(
                GraphConvBlock('drug', input_dim=last, output_dim=dim, adj=self.drug_code_adj).to(device))
            last = dim
        last = code_dim
        for layer, dim in enumerate(code_hidden_dims):
            self.code_blocks.append(
                GraphConvBlock('code', input_dim=last, output_dim=dim, adj=self.code_drug_adj).to(device))
            last = dim


        c2d_dims = ([drug_dim] + drug_hidden_dims)[:-1]

        d2c_dims = ([code_dim] + code_hidden_dims)[:-1]
        self.c2d_linears = [MyLinear(dim) for layer, dim in enumerate(c2d_dims)]
        self.d2c_linears = [MyLinear(dim) for layer, dim in enumerate(d2c_dims)]

    def forward(self, drug_embeddings, code_embeddings):
        weight = norm_no_nan(self.code_code_adj)
        for c2d_linear, d2c_linear, drug_block, code_block in zip(self.c2d_linears, self.d2c_linears, self.drug_blocks,
                                                                  self.code_blocks):
 
            code_embedding_d = c2d_linear(code_embeddings)

            drug_embeddings_new = drug_block(drug_embeddings, code_embedding_d)

            drug_embeddings_c = d2c_linear(drug_embeddings)

            code_embeddings = code_block(code_embeddings, drug_embeddings_c, weight)
            drug_embeddings = drug_embeddings_new

        drug_embeddings_c = self.d2c_linears[-1](drug_embeddings)
        code_embeddings = self.code_blocks[-1](code_embeddings, drug_embeddings_c, weight)
        return drug_embeddings, code_embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.alpha = 0.01
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)

        x = x.view(*new_x_shape)
        return x

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
 
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        return outputs

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size,intermediate_size):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.dense_dag = nn.Linear(hidden_size, intermediate_size)
        self.linear = nn.Linear(intermediate_size,hidden_size)

    def forward(self, hidden_states, hidden_states_dag):
        hidden_states_ = self.dense(hidden_states)
        hidden_states_dag_ = self.dense_dag(hidden_states_dag)
        hidden_states = self.linear(F.relu(hidden_states_+hidden_states_dag_))
        return hidden_states

class VisitEmbedding(nn.Module):
    def __init__(self, max_visit_len):
        super().__init__()
        self.max_seq_len = max_visit_len

    def forward(self, code_embeddings, visit_codes, visit_lens):

        visit_codes_embedding = F.embedding(visit_codes, code_embeddings)
        visit_codes_mask = torch.unsqueeze(visit_codes > 0, dim=-1).to(dtype=visit_codes_embedding.dtype,
                                                                       device=visit_codes.device)
        visit_codes_embedding *= visit_codes_mask

        visit_codes_num = torch.unsqueeze(
            torch.sum((visit_codes > 0).to(dtype=visit_codes_embedding.dtype), dim=-1), dim=-1)

        sum_visit_codes_embedding = torch.sum(visit_codes_embedding, dim=-2)
        visit_codes_num[visit_codes_num == 0] = 1

        visits_embeddings = sum_visit_codes_embedding / visit_codes_num

        visit_mask = seq_mask(visit_lens, self.max_seq_len).unsqueeze(-1).to(dtype=visits_embeddings.dtype)
        visits_embeddings *= visit_mask
        return visits_embeddings


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, ffn_dim=1024, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim=model_dim, ffn_dim=ffn_dim, dropout=dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


def padding_mask(seq_q, seq_k):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1) 
    return pad_mask


class Encoder(nn.Module):
    def __init__(self, max_visit_len, num_layers, model_dim, num_heads, ffn_dim, time_dim,
                 dropout,device):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim=model_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in
            range(num_layers)
        ])
        self.selection_layer = nn.Linear(1, time_dim)
 
        self.time_layer = nn.Linear(time_dim, model_dim)
        self.pos_embedding = PositionalEncoding(model_dim=model_dim, max_visit_len=max_visit_len)
        self.tanh = nn.Tanh()

        self.odeNet_los = ODENet(device, model_dim, model_dim)
        self.odeNet_interval = ODENet(device, model_dim, model_dim,
                                      output_dim=model_dim, augment_dim=10, time_dependent=True)
        self.LayerNorm = BertLayerNorm(model_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.2)

    def forward(self, visits_embeddings, intervals, visit_mask, visit_lens, mark):
        v_mask = visit_mask.unsqueeze(-1)

        intervals = intervals.unsqueeze(-1) / 180

        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(intervals), 2))

        time_feature = self.time_layer(time_feature)
        output_pos, ind_pos = self.pos_embedding(visit_lens.unsqueeze(-1))

        seq_length = visits_embeddings.size(1)
        integration_time = torch.linspace(0., 1., seq_length)
        y0 = visits_embeddings[:, 0, :]
        interval_embeddings = self.odeNet_interval(y0, integration_time).permute(1, 0, 2)
        los_embeddings = self.odeNet_los(visits_embeddings)
        output = visits_embeddings + time_feature + output_pos + interval_embeddings + los_embeddings
        output *= v_mask
        att_mask = padding_mask(ind_pos, ind_pos)
        for encoder in self.encoder_layers:
            output, _ = encoder(output, att_mask)

        return output


class Attention(nn.Module):
    def __init__(self, input_size, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.u_omega = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(attention_dim, 1))))
        self.w_omega = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(input_size, attention_dim))))

    def forward(self, x, mask=None):
        """
            x: (batch_size, max_seq_len, rnn_dim[-1])
        """
        t = torch.matmul(x, self.w_omega)

        vu = torch.squeeze(torch.tensordot(t, self.u_omega, dims=1), dim=-1)
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = F.softmax(vu)

        output = torch.sum(x * alphas.unsqueeze(-1), dim=-2)
        return output, alphas


class TimeEncoder(nn.Module):
    def __init__(self, time_dim, quiry_dim):
        super(TimeEncoder, self).__init__()
        self.selection_layer = nn.Linear(1, time_dim)
        self.weight_layer = nn.Linear(time_dim, quiry_dim)
        self.quiry_dim = quiry_dim
        self.tanh = nn.Tanh()
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, intervals, final_queries, mask_mult):
        intervals = torch.unsqueeze(intervals, dim=2) / 180
        selection_features = 1 - self.tanh(torch.pow(self.selection_layer(intervals), 2))
        selection_features = self.leakyRelu(self.weight_layer(selection_features))
        r = torch.bmm(final_queries, selection_features.transpose(-1, -2)).squeeze()
        selection_features = r / math.sqrt(self.quiry_dim)
        selection_features = selection_features.masked_fill_(mask_mult, -torch.inf)
        return F.softmax(selection_features, dim=1)


def log_no_nan(x):
    mask = (x == 0).to(dtype=x.dtype, device=x.device)
    x = x + mask
    return torch.log(x)


def div_no_nan(x, y):
    mask = (y == 0).to(dtype=x.dtype, device=x.device)
    y = y + mask
    return torch.div(x, y)


class WordEmbedding(nn.Module):
    def __init__(self, dictionary: OrderedDict, word_dim):
        super().__init__()
        word_list = list(dictionary.values())
        fasttext_model = FastText([word_list], vector_size=word_dim, window=3, min_count=1, sg=1, epochs=20)
        embedding_matrix = []
        for word in word_list:
            embedding_matrix.append(fasttext_model.wv[word])
        embedding_matrix = np.array(embedding_matrix)
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        zero = torch.zeros((1, word_dim), dtype=embedding_matrix.dtype)
        embeddings = torch.cat([zero, embedding_matrix], dim=0)
        self.embeddings = nn.Embedding(len(dictionary) + 1, word_dim)
        self.embeddings.weight = nn.Parameter(embeddings, requires_grad=True)

    def forward(self, inputs):
        return self.embeddings(inputs)


class NoteAttention(nn.Module):
    def __init__(self, input_size, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.w_omega = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(size=(input_size, self.attention_dim))))

    def forward(self, x, ctx_vector, mask=None):
        t = torch.matmul(x, self.w_omega)
        vu = torch.sum(t * torch.unsqueeze(ctx_vector, dim=1), dim=-1)
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = F.softmax(vu) 
        output = torch.sum(t * torch.unsqueeze(alphas, -1), dim=-2)
        return output, alphas


class NoteEmbedding(nn.Module):
    def __init__(self, word_dim, attention_dim, outputdim, max_seq_len):
        super().__init__()
        self.attention = NoteAttention(word_dim, attention_dim)
        self.max_seq_len = max_seq_len
        # self.linear = nn.Linear(attention_dim, outputdim)
        self.linear = KAN(attention_dim, outputdim)

    def forward(self, word_embeddings, word_lens, visit_output):
        word_mask = seq_mask(word_lens, self.max_seq_len).to(dtype=word_embeddings.dtype)
        word_embeddings *= word_mask.unsqueeze(-1)
        outputs, weight = self.attention(word_embeddings, visit_output, word_mask)
        outputs = self.linear(outputs)
        return outputs


class DrugNoteEmbedding(nn.Module):
    def __init__(self, word_dim, gru_dims, output_dim):
        super(DrugNoteEmbedding, self).__init__()
        use_bi = True
        self.gru = nn.GRU(input_size=word_dim, hidden_size=gru_dims[-1], num_layers=len(gru_dims), batch_first=True,
                          bidirectional=use_bi)
        if not use_bi:
            in_feature = gru_dims[-1]
        else:
            in_feature = 2 * gru_dims[-1]
        self.linear = nn.Linear(in_features=in_feature, out_features=output_dim)
        self.leakyrelu = nn.LeakyReLU()
        self.u = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(output_dim, 1))))
        self.lambda_ = 0.3

    def forward(self, word_embeddings, mask, tf_idf=None, training=False):
        outputs, _ = self.gru(word_embeddings)
        outputs = self.linear(outputs)
        t = self.leakyrelu(outputs)
        uz = torch.tensordot(t, self.u, dims=1).squeeze()
        uz *= mask
        weight = masked_softmax(uz, mask)
        outputs = torch.sum(outputs * weight.unsqueeze(-1), dim=-2)
        if training:
            loss = log_no_nan(div_no_nan(weight, tf_idf))
            loss = torch.sum(torch.sum(weight * loss, dim=-1))
            return outputs, loss
        else:
            return outputs
