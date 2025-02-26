import os
import random
import time
from tqdm import tqdm
import torch
import numpy as np
import _pickle as pickle

from metrics import evaluate_codes, evaluate_hf
from models.model import Model
from utils import EHRDataset, format_time, medical_codes_loss, MultiStepLRScheduler
import torch.optim as optim


def historical_hot(code_x, code_num):
    result = np.zeros((len(code_x), code_num), dtype=int)

    for i, x in enumerate(code_x):
        for code in x:

            result[i][code - 1] = 1
    return result


def load_data(encoded_path):
    code_map = pickle.load(open(os.path.join(encoded_path, 'code_map.pkl'), 'rb'))
    drug_map = pickle.load(open(os.path.join(encoded_path, 'drug_map.pkl'), 'rb'))
    dictionary = pickle.load(
        open(os.path.join(encoded_path, 'dictionary.pkl'), 'rb'))
    d_dictionary = pickle.load(
        open(os.path.join(encoded_path, 'd_dictionary.pkl'), 'rb'))
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    return code_map, drug_map, dictionary, d_dictionary, auxiliary


if __name__ == '__main__':
    dataset = 'mimic3'
    data_path = os.path.join('data', dataset)
    encoded_path = os.path.join(data_path, 'encoded')
    standard_path = os.path.join(data_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    seed = 18
    task = 'h'  # m or h
    check = (task == 'm' or task == 'h')
    assert check
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f"device:{device}")
    batch_size = 32
    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    code_map, drug_map, dictionary, d_dictionary, auxiliary = load_data(encoded_path)
    code_code_adj, drug_code_adj, code_levels = auxiliary['code_code_adj'], auxiliary['drug_code_adj'], auxiliary[
        'code_levels']

    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, device=device, batch_size=batch_size, shuffle=True)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, device=device, batch_size=batch_size, shuffle=False)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, device=device, batch_size=batch_size, shuffle=False)

    code_num = len(code_map)

    drug_num = len(drug_map)

    if task == 'm':
        if dataset == 'mimic3' or dataset == 'eicu':
            code_dims = [32] * 4
        else:
            code_dims = [64] * 4
    else:
        if dataset == 'mimic3' or dataset == 'eicu':
            code_dims = [7] * 4
        else:
            code_dims = [5] * 4

    use_pretrain = False
    if use_pretrain and task == 'm':
        embedding_init = pickle.load(
            open('./saved/hyperbolic/%s_%s Model_leaf_embeddings' % (dataset, 'Klein'), 'rb'))
        pre_embedding = torch.tensor(embedding_init[:len(code_map) + 1]).to(dtype=torch.float32, device=device)
    else:
        pre_embedding = torch.empty(len(code_map) + 1, np.sum(code_dims))
        torch.nn.init.xavier_uniform_(pre_embedding)
        pre_embedding.to(dtype=torch.float32, device=device)
    config = {
        'use_pretrain': use_pretrain,
        'drug_code_adj': torch.tensor(drug_code_adj, dtype=torch.float32, device=device),
        'code_code_adj': torch.tensor(code_code_adj, dtype=torch.float32, device=device),
        'code_levels': torch.tensor(code_levels, dtype=torch.int32, device=device),
        'code_num_in_levels': np.max(code_levels, axis=0) + 1,
        'code_num': code_num,
        'drug_num': drug_num,
        'max_visit_seq_len': train_data.code_x.shape[1],
        'lambda': 0.3,
        'device': device,
        'dictionary': dictionary,
        'd_dictionary': d_dictionary,
        'max_disease_seq_len': train_data.disease_x.shape[1],
        'pre_embedding': pre_embedding
    }

    attention_dim = 64
    hyper_params = {
        'code_dims': code_dims,
        'drug_dim': 64 if task == 'm' else 16,
        'drug_hidden_dims': [64] if task == 'm' else [16],
        'code_hidden_dims': [64, np.sum(code_dims)] if task == 'm' else [10, np.sum(code_dims)],
        'intermediate_dims': 512,
        'input_dim': np.sum(code_dims),
        'quiry_dim': 32 if task == 'm' else 16,
        'time_dim': 32 if task == 'm' else 16,
        'attention_dim': attention_dim,
        'num_layers': 2 if task == 'm' else 1,
        'num_heads': 4,
        'ffn_dim': 1024,
        'word_dim': 16,
        'disease_outdim': attention_dim,
        'drug_word_dim': 16,
        'drug_outdim': attention_dim,
        'gru_dims': [64]
    }
    hyper_params['input_dim'] = hyper_params['input_dim'] + hyper_params['disease_outdim'] + hyper_params['drug_outdim']


    def lr_schedule_fn(epoch):
        if epoch < 20:
            return 0.1
        elif epoch < 100:
            return 0.01
        elif epoch < 200:
            return 0.001
        else:
            return 0.0001


    task_conf = {
        'm': {
            'dropout': 0.3,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'epochs': 70,
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'epochs': 100,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }
    epochs = task_conf[task]['epochs']
    output_size = task_conf[task]['output_size']
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']
    hyper_params['output_dim'] = output_size
    hyper_params['dropout'] = dropout_rate

    model = Model(config, hyper_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    if task == 'm':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_fn)
    else:
        scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                         task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fn = medical_codes_loss

    test_historical = historical_hot(valid_data.code_x, code_num)
    print(f"params num: {pytorch_total_params}")
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        loss = 0
        steps = len(train_data)
        st = time.time()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            visit_codes, visit_lens, intervals, disease_x, disease_lens, drug_x, drug_lens, mark, y, tf_idf_weight = \
                train_data[step]
            output, drug_loss = model(visit_codes, visit_lens, intervals, disease_x, disease_lens, drug_x, drug_lens,
                                      mark, tf_idf_weight, True)
            loss = loss_fn(output, y) + drug_loss
            loss.backward()
            optimizer.step()
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, loss), end='')
            
        scheduler.step()
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, loss))
        evaluate_fn(model, valid_data, loss_fn, test_historical)

    print("Test data:\n")
    evaluate_fn(model, test_data, loss_fn, test_historical)

    
