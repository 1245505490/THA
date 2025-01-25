import numpy as np
import torch
from torch import nn

from models.kan import KAN, NaiveFourierKANLayer
from models.layers import *
from models.utils import seq_mask, final_mask, get_multmask


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        # self.linear = KAN(input_size, output_size)
        # self.linear = NaiveFourierKANLayer(input_size, output_size)
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, training=True):
        output = self.dropout(x)
        output = self.linear(output)
        return output


class Model(nn.Module):
    def __init__(self, config, hyper_params):
        super().__init__()
        device = config['device']
        self.cgl = CGLFeatureExtractor(config, hyper_params).to(device)
        self.classifier = Classifier(input_size=hyper_params['input_dim'], output_size=hyper_params['output_dim'],
                                     dropout_rate=hyper_params['dropout']).to(device)

    def forward(self, visit_codes, visit_lens, intervals, disease_x, disease_lens, drug_x, drug_lens, mark,
                tf_idf_weight=None, training=False):
        inputs = {
            'visit_codes': visit_codes,
            'visit_lens': visit_lens,
            'intervals': intervals,
            'disease_x': disease_x,
            'disease_lens': disease_lens,
            'drug_x': drug_x,
            'drug_lens': drug_lens,
            'mark': mark,
            'tf_idf': tf_idf_weight,
            'training': training
        }
        if training:
            output, drug_loss = self.cgl(inputs)
            output = self.classifier(output, training)
            return output, drug_loss
        else:
            output = self.cgl(inputs)
            output = self.classifier(output, training)
            return output


class CGLFeatureExtractor(nn.Module):
    def __init__(self, config, hyper_params):
        super().__init__()
        self.config = config
        self.hyper_params = hyper_params
        self.device = config['device']
        self.hierarchical_embedding_layer = HierarchicalEmbedding(code_levels=config['code_levels'],
                                                                  code_num_in_levels=config['code_num_in_levels'],
                                                                  code_dims=hyper_params['code_dims'],
                                                                  code_num=config['code_num'],
                                                                  ).to(self.device)
        self.pre_embedding = config['pre_embedding']
        self.drug_embedding_layer = DrugEmbedding(drug_num=config['drug_num'], drug_dim=hyper_params['drug_dim']).to(
            self.device)
        code_dim = np.sum(hyper_params['code_dims'])
        drug_dim = hyper_params['drug_dim']
        self.max_visit_len = config['max_visit_seq_len']
        self.gcn = GraphConvolution(drug_dim=drug_dim, code_dim=code_dim, drug_code_adj=config['drug_code_adj'],
                                    code_code_adj=config['code_code_adj'],
                                    drug_hidden_dims=hyper_params['drug_hidden_dims'],
                                    code_hidden_dims=hyper_params['code_hidden_dims'], device=self.device).to(
            self.device)
        self.visit_embedding_layer = VisitEmbedding(max_visit_len=self.max_visit_len).to(self.device)
        self.feature_encoder = Encoder(max_visit_len=config['max_visit_seq_len'],
                                       num_layers=hyper_params['num_layers'], model_dim=code_dim,
                                       num_heads=hyper_params['num_heads'], ffn_dim=hyper_params['ffn_dim'],
                                       time_dim=hyper_params['time_dim'],
                                       dropout=hyper_params['dropout'],device=self.device).to(self.device)
        self.quiry_layer = nn.Linear(code_dim, hyper_params['quiry_dim'])
        self.time_encoder = TimeEncoder(time_dim=hyper_params['time_dim'], quiry_dim=hyper_params['quiry_dim']).to(
            self.device)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.quiry_weight_layer = nn.Linear(code_dim, 2)
        self.quiry_weight_layer2 = nn.Linear(code_dim, 1)
        self.attention = Attention(code_dim, attention_dim=hyper_params['attention_dim']).to(self.device)
        self.word_embedding_layer = WordEmbedding(config['dictionary'], hyper_params['word_dim'])
        self.note_embedding_layer = NoteEmbedding(hyper_params['word_dim'], int(code_dim),
                                                  hyper_params['disease_outdim'],
                                                  config['max_disease_seq_len'])
        self.drugword_embedding_layer = WordEmbedding(config['d_dictionary'], hyper_params['drug_word_dim'])
        self.drug_note_embedding_layer = DrugNoteEmbedding(hyper_params['drug_word_dim'], hyper_params['gru_dims'],
                                                           hyper_params['drug_outdim'])
        self.MultiHead1 = BertSelfAttention(code_dim, hyper_params['num_heads'])
        self.MultiHead2 = BertSelfAttention(code_dim, hyper_params['num_heads'])
        self.intermediate = BertIntermediate(code_dim, hyper_params['intermediate_dims'])

    def forward(self, inputs):
        visit_codes = inputs['visit_codes']
        visit_lens = inputs['visit_lens']

        intervals = inputs['intervals']
        word_ids = inputs['disease_x']
        word_lens = torch.reshape(inputs['disease_lens'], (-1,))
        drug_word_ids = inputs['drug_x']
        drug_lens = torch.reshape(inputs['drug_lens'], (-1,))
        mark = inputs['mark']
        word_tf_idf = inputs['tf_idf']
        training = inputs['training']

        visit_mask = seq_mask(visit_lens, self.max_visit_len)
        mask_final = final_mask(visit_lens, self.max_visit_len)
        mask_mult = get_multmask(visit_mask)
  
        hierarchical_embeddings = self.hierarchical_embedding_layer()
  
        hyperbolic_embeddings = self.pre_embedding

        drug_embeddings = self.drug_embedding_layer()

        hierarchical_multi = self.MultiHead1(hierarchical_embeddings)
        hyperbolic_multi = self.MultiHead2(hyperbolic_embeddings)
        Ht = self.intermediate(hierarchical_multi, hyperbolic_multi)
        code_embeddings = F.layer_norm(Ht + hierarchical_multi + hyperbolic_multi,
                                       normalized_shape=(hierarchical_embeddings.size(-1),))
        drug_embeddings, code_embeddings = self.gcn(drug_embeddings=drug_embeddings, code_embeddings=code_embeddings)
        visits_embeddings = self.visit_embedding_layer(code_embeddings, visit_codes, visit_lens)

        features = self.feature_encoder(visits_embeddings, intervals, visit_mask, visit_lens, mark)

        final_statues = features * mask_final.unsqueeze(-1)

        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.leakyRelu(self.quiry_layer(final_statues))
    
        _, self_weight = self.attention(features, visit_mask)
        self_weight = self_weight.unsqueeze(-1)

        time_weight = self.time_encoder(intervals, quiryes, mask_mult).unsqueeze(-1)
    
        attention_weight = torch.softmax(self.quiry_weight_layer2(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
   
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)

        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-8)

        self.total_weight = total_weight
        weighted_features = features * total_weight

        output = torch.sum(weighted_features, 1)

        word_embeddings = self.word_embedding_layer(word_ids)
        word_output = self.note_embedding_layer(word_embeddings, word_lens, output)

        drugText_embeddings = self.drugword_embedding_layer(drug_word_ids)
        drug_mask = (drug_word_ids > 0).to(dtype=drugText_embeddings.dtype)
        if training:
            drug_output, drug_loss = self.drug_note_embedding_layer(drugText_embeddings, drug_mask, word_tf_idf,
                                                                    training)
            outputs = torch.cat([output, word_output, drug_output], dim=-1)
            return outputs, drug_loss
        else:
            drug_output = self.drug_note_embedding_layer(drugText_embeddings, drug_mask)
            outputs = torch.cat([output, word_output, drug_output], dim=-1)
            return outputs
