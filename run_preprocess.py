import os
import _pickle as pickle

from preprocess.auxiliary import *
from preprocess.build_dataset import split_patients, build_code_xy, build_tf_idf_weight, \
    build_heart_failure_y, build_disease_x, build_drug_x, build_code_xy_eicu
from preprocess.parse_csv import Mimic3Parser, Mimic4Parser, EICUParser
from preprocess.encode import encode_code, encode_drug, encode_train_disease, encode_other_disease, encode_train_drug, \
    encode_other_drug
from preprocess import save_sparse, save_data

if __name__ == '__main__':
    conf = {
        'mimic3': {
            'parser': Mimic3Parser,
            'train_num': 6000,
            'valid_num': 500,
            'threshold': 0.01
        },
        'mimic4': {
            'parser': Mimic4Parser,
            'train_num': 8000,
            'valid_num': 1000,
            'threshold': 0.01,
            'sample_num': 10000
        },
        'eicu': {
            'parser': EICUParser,
            'train_num': 8000,
            'valid_num': 1000,
            'threshold': 0.01,
        },
    }
    from_saved = True
    data_path = 'data'
    dataset = 'mimic3'
    seed = 18

    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    standard_path = os.path.join(dataset_path, 'standard')
    parsed_path = os.path.join(dataset_path, 'parsed')
    encoded_path = os.path.join(dataset_path, 'encoded')
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    if not os.path.exists(raw_path):
        print('please put the CSV files in `data/%s/raw`' % dataset)
        create_dir(raw_path)
        exit()
    create_dir(standard_path)
    create_dir(parsed_path)
    create_dir(encoded_path)

    patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
    admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
    admission_drugs = pickle.load(open(os.path.join(parsed_path, 'admission_drugs.pkl'), 'rb'))
    patient_disease = pickle.load(open(os.path.join(parsed_path, 'patient_disease.pkl'), 'rb'))
    patient_drug = pickle.load(open(os.path.join(parsed_path, 'patient_drug.pkl'), 'rb'))
    

    patient_num = len(patient_admission)
    
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])

    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
 
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
  
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)

    max_visit_drug_num = max([len(codes) for codes in admission_drugs.values()])
    avg_visit_drug_num = sum([len(codes) for codes in admission_drugs.values()]) / len(admission_drugs)

    print('patient num: %d' % patient_num)
    print('max admission num: %d' % max_admission_num)
    print('mean admission num: %.2f' % avg_admission_num)
    print('max code num in an admission: %d' % max_visit_code_num)
    print('mean code num in an admission: %.2f' % avg_visit_code_num)
    print('max drug num in an admission: %d' % max_visit_drug_num)
    print('mean drug num in an admission: %.2f' % avg_visit_drug_num)
    print('encoding code ...')
  
    admission_codes_encoded, code_map = encode_code(patient_admission, admission_codes)
    admission_drugs_encoded, drug_map = encode_drug(patient_admission, admission_drugs)

    code_num = len(code_map)
    print('There are %d codes' % code_num)
    drug_num = len(drug_map)
    print('There are %d drugs' % drug_num)

    train_pids, valid_pids, test_pids = split_patients(
        patient_admission=patient_admission,
        admission_codes=admission_codes,
        code_map=code_map,
        train_num=conf[dataset]['train_num'],
        valid_num=conf[dataset]['valid_num']
    )
    print('There are %d train, %d valid, %d test samples' % (len(train_pids), len(valid_pids), len(test_pids)))

    train_disease_encoded, dictionary = encode_train_disease(patient_disease, train_pids)
    valid_disease_encoded = encode_other_disease(patient_disease, valid_pids, dictionary)
    test_disease_encoded = encode_other_disease(patient_disease, test_pids, dictionary)
    all_disease_data = (train_disease_encoded, valid_disease_encoded, test_disease_encoded)
    pickle.dump(all_disease_data, open('all_disease_data_' + dataset + '.pkl', 'wb'))
    train_drug_encoded, d_dictionary = encode_train_drug(patient_drug, train_pids)
    valid_drug_encoded = encode_other_drug(patient_drug, valid_pids, d_dictionary)
    test_drug_encoded = encode_other_drug(patient_drug, test_pids, d_dictionary)

    pickle.dump(dictionary, open(os.path.join(encoded_path, 'dictionary.pkl'), 'wb'))
    pickle.dump(d_dictionary, open(os.path.join(encoded_path, 'd_dictionary.pkl'), 'wb'))
    code_code_adj = generate_code_code_adjacent(train_pids, patient_admission, admission_codes_encoded, code_num)
    drug_code_adj = generate_drug_code_adjacent(train_pids, patient_admission, admission_codes_encoded,
                                                admission_drugs_encoded, code_num, drug_num)


    def max_word_num(encoded: dict) -> int:
        return max(len(note) for note in encoded.values())


    max_word_num_in_a_disease_text = max(
        [max_word_num(train_disease_encoded), max_word_num(valid_disease_encoded), max_word_num(test_disease_encoded)])
    print('max word num in a disease text:', max_word_num_in_a_disease_text)

    max_word_num_in_a_drug_text = max(
        [max_word_num(train_drug_encoded), max_word_num(valid_drug_encoded), max_word_num(test_drug_encoded)])
    print('max word num in a drug text:', max_word_num_in_a_drug_text)

    common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num, max_visit_code_num]
    if dataset != 'eicu':
        (train_code_x, train_codes_y, train_visit_lens, train_intervals, train_mark) = build_code_xy(train_pids,
                                                                                                     *common_args)
        (valid_code_x, valid_codes_y, valid_visit_lens, valid_intervals, valid_mark) = build_code_xy(valid_pids,
                                                                                                     *common_args)
        (test_code_x, test_codes_y, test_visit_lens, test_intervals, test_mark) = build_code_xy(test_pids,
                                                                                                *common_args)
    else:
        (train_code_x, train_codes_y, train_visit_lens, train_intervals, train_mark) = build_code_xy_eicu(train_pids,
                                                                                                          *common_args)
        (valid_code_x, valid_codes_y, valid_visit_lens, valid_intervals, valid_mark) = build_code_xy_eicu(valid_pids,
                                                                                                          *common_args)
        (test_code_x, test_codes_y, test_visit_lens, test_intervals, test_mark) = build_code_xy_eicu(test_pids,
                                                                                                     *common_args)
    train_disease_x, train_disease_lens = build_disease_x(train_pids, train_disease_encoded,
                                                          max_word_num_in_a_disease_text)
    valid_disease_x, valid_disease_lens = build_disease_x(valid_pids, valid_disease_encoded,
                                                          max_word_num_in_a_disease_text)
    test_disease_x, test_disease_lens = build_disease_x(test_pids, test_disease_encoded, max_word_num_in_a_disease_text)

    train_disease_data = (train_disease_x, train_disease_lens)
    valid_disease_data = (valid_disease_x, valid_disease_lens)
    test_disease_data = (test_disease_x, test_disease_lens)

    train_drug_x, train_drug_lens = build_drug_x(train_pids, train_drug_encoded, max_word_num_in_a_drug_text)
    valid_drug_x, valid_drug_lens = build_drug_x(valid_pids, valid_drug_encoded, max_word_num_in_a_drug_text)
    test_drug_x, test_drug_lens = build_drug_x(test_pids, test_drug_encoded, max_word_num_in_a_drug_text)

    tf_idf_weight = build_tf_idf_weight(train_pids, train_drug_x, train_drug_encoded, len(d_dictionary))
    train_drug_data = (train_drug_x, train_drug_lens, tf_idf_weight)
    valid_drug_data = (valid_drug_x, valid_drug_lens)
    test_drug_data = (test_drug_x, test_drug_lens)

    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

    code_levels = generate_code_levels(data_path, code_map)


    pickle.dump({
        'code_levels': code_levels,
        'drug_code_adj': drug_code_adj,
        'code_code_adj': code_code_adj
    }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))

    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump(admission_drugs_encoded, open(os.path.join(encoded_path, 'drugs_encoded.pkl'), 'wb'))
    pickle.dump(drug_map, open(os.path.join(encoded_path, 'drug_map.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)
    print('\tsaving training data')
    save_data(train_path, train_code_x, train_visit_lens, train_codes_y, train_hf_y, train_intervals, train_mark)
    pickle.dump(train_disease_data, open(os.path.join(train_path, 'disease_dataset.pkl'), 'wb'))
    pickle.dump(train_drug_data, open(os.path.join(train_path, 'drug_dataset.pkl'), 'wb'))

    print('\tsaving valid data')
    save_data(valid_path, valid_code_x, valid_visit_lens, valid_codes_y, valid_hf_y, valid_intervals, valid_mark)
    pickle.dump(valid_disease_data, open(os.path.join(valid_path, 'disease_dataset.pkl'), 'wb'))
    pickle.dump(valid_drug_data, open(os.path.join(valid_path, 'drug_dataset.pkl'), 'wb'))

    print('\tsaving test data')
    save_data(test_path, test_code_x, test_visit_lens, test_codes_y, test_hf_y, test_intervals, test_mark)
    pickle.dump(test_disease_data, open(os.path.join(test_path, 'disease_dataset.pkl'), 'wb'))
    pickle.dump(test_drug_data, open(os.path.join(test_path, 'drug_dataset.pkl'), 'wb'))
