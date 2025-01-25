from datetime import datetime
import math

import numpy as np


def split_patients(patient_admission, admission_codes, code_map, train_num, valid_num, seed=18):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission['adm_id']]
 
                if code in codes:
                    common_pids.add(pid)
                    break

            else:
                continue
            break
    print('\r\t100%')

    max_admission_num = 0

    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid

    common_pids.add(pid_max_admission_num)

    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num,
                  max_code_num_in_a_visit):
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_code_num_in_a_visit), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    times = []
    mark = np.zeros((n, max_admission_num, 4), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        t = [0] * max_admission_num
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['adm_id']]
            x[i, k, :len(codes)] = codes
            t[k] = datetime.fromtimestamp(datetime.timestamp(admission['adm_time']))
            time = admission['adm_time']
            mark[i][k][0] = time.year
            mark[i][k][1] = time.month
            mark[i][k][2] = time.day
            mark[i][k][3] = time.hour
        codes = np.array(admission_codes_encoded[admissions[-1]['adm_time']]) - 1
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
        times.append(t)
    times = np.array(times)
    intervals = np.zeros_like(times, dtype=float)
    for i in range(len(times)):
        for j in range(lens[i]):
            if j > 0:
                intervals[i][j] = (times[i][j] - times[i][j - 1]).days
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens, intervals, mark

def build_code_xy_eicu(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num,
                  max_code_num_in_a_visit):
    n = len(pids)
    x = np.zeros((n, max_admission_num, max_code_num_in_a_visit), dtype=int)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    times = []
    mark = np.zeros((n, max_admission_num, 4), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        t = [0] * max_admission_num
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['adm_id']]
            x[i, k, :len(codes)] = codes
            t[k] = admission['adm_time']
        codes = np.array(admission_codes_encoded[admissions[-1]['adm_id']]) - 1
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
        times.append(t)
    times = np.array(times)
    intervals = np.zeros_like(times, dtype=float)
    for i in range(len(times)):
        for j in range(lens[i]):
            if j > 0:
                intervals[i][j] = (times[i][j] - times[i][j - 1])
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens, intervals, mark

def build_heart_failure_y(hf_prefix: str, codes_y: np.ndarray, code_map: dict) -> np.ndarray:
    print('building train/valid/test heart failure labels ...')

    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)

    hfs[hf_list - 1] = 1

    hf_exist = np.logical_and(codes_y, hfs)

    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y


def build_disease_x(pids: np.ndarray,
                    disease_encoded: dict,
                    max_word_num_in_a_note: int) -> (np.ndarray, np.ndarray):
    print('building train/valid/test disease text features...')
    n = len(pids)
    x = np.zeros((n, max_word_num_in_a_note), dtype=int)
    lens = np.zeros((n,), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        disease = disease_encoded[pid]
        length = max_word_num_in_a_note if max_word_num_in_a_note < len(disease) else len(disease)
        x[i][:length] = disease[:length]
        lens[i] = length
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, lens


def build_drug_x(pids: np.ndarray,
                 drug_encoded: dict,
                 max_word_num_in_a_note: int) -> (np.ndarray, np.ndarray):
    print('building train/valid/test drug text features...')
    n = len(pids)

    x = np.zeros((n, max_word_num_in_a_note), dtype=int)
    lens = np.zeros((n,), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        disease = drug_encoded[pid]
        length = max_word_num_in_a_note if max_word_num_in_a_note < len(disease) else len(disease)
        x[i][:length] = disease[:length]
        lens[i] = length
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, lens


def calculate_tf_idf(note_encoded: dict, word_num: int) -> dict:

    n_docs = len(note_encoded)
    tf = dict()
    df = np.zeros((word_num + 1,), dtype=np.int64)
    print('calculating tf and df ...')
    for i, (pid, note) in enumerate(note_encoded.items()):
        print('\r\t%d / %d' % (i + 1, n_docs), end='')
        note_tf = dict()
        for word in note:

            note_tf[word] = note_tf.get(word, 0) + 1
        wset = set(note)
        for word in wset:

            df[word] += 1
        tf[pid] = note_tf
    print('\r\t%d / %d patients' % (n_docs, n_docs))
    print('calculating tf_idf ...')
    tf_idf = dict()
    for i, (pid, note) in enumerate(note_encoded.items()):
        print('\r\t%d / %d patients' % (i + 1, n_docs), end='')
        note_tf = tf[pid]

        note_tf_idf = [note_tf[word] / len(note) * (math.log(n_docs / (1 + df[word]), 10) + 1)
                       for word in note]
        tf_idf[pid] = note_tf_idf
    print('\r\t%d / %d patients' % (n_docs, n_docs))
    return tf_idf


def build_tf_idf_weight(pids: np.ndarray, note_x: np.ndarray, note_encoded: dict, word_num: int) -> np.ndarray:
    print('build tf_idf for notes ...')
    tf_idf = calculate_tf_idf(note_encoded, word_num)

    weight = np.zeros_like(note_x, dtype=float)
    for i, pid in enumerate(pids):
        note_tf_idf = tf_idf[pid]
        weight[i][:len(note_tf_idf)] = note_tf_idf

    weight = weight / weight.sum(axis=-1, keepdims=True)
    return weight
