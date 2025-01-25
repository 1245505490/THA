import copy
from collections import OrderedDict
import re
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))


def encode_code(patient_admission, admission_codes):
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        if len(admissions) <= 1:
            continue
        for admission in admissions:
            codes = admission_codes[admission['adm_id']]
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map) + 1

    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes))
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map


def encode_drug(patient_admission, admission_drugs):
    drug_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            hadm = admission['adm_id']
            if hadm not in admission_drugs:
                continue
            drugs = admission_drugs[hadm]
            for drug in drugs:
                if drug not in drug_map:
                    drug_map[drug] = len(drug_map)

    admission_drugs_encoded = {
        admission_id: list(set(drug_map[drug] for drug in drugs))
        for admission_id, drugs in admission_drugs.items()
    }
    return admission_drugs_encoded, drug_map


def encode_train_disease(patient_disease: dict, pids: np.ndarray):
    print('encoding train disease text...')
    dictionary = OrderedDict()
    disease_encoded = dict()
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        words = extract_word(patient_disease[pid])
        encoded = []
        for word in words:
            if word not in dictionary:
                wid = len(dictionary) + 1
                dictionary[word] = wid
            else:
                wid = dictionary[word]
            encoded.append(wid)
        disease_encoded[pid] = encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return disease_encoded, dictionary


def encode_other_disease(patient_disease: dict, pids: np.ndarray, dictionary: dict):
    print('encoding valid/test disease text...')
    disease_encoded = dict()
    for i, pid in enumerate(pids):
        words = extract_word(patient_disease[pid])
        encoded = []
        for word in words:
            if word in dictionary:
                encoded.append(dictionary[word])
        if len(encoded) == 0:
            encoded.append(0)
        disease_encoded[pid] = encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return disease_encoded


def extract_word(text: str) -> list:
    text = re.sub(r'[^A-Za-z0-9%*_]', ' ', text.strip().lower())
    words = word_tokenize(text)
    clean_words = []
    for word in words:
        if word not in stopwords_set:
            word = ps.stem(word).lower()
            if word not in stopwords_set:
                clean_words.append(word)

    return clean_words


def encode_train_drug(patient_drug: OrderedDict, pids: np.ndarray):
    print('encoding train drug text ...')
    dictionary = OrderedDict()
    drug_encoded = dict()
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        encoded = []
        if pid not in patient_drug:
            drug_encoded[pid] = [0]
            continue
        words = extract_word(patient_drug[pid])
        for word in words:
            if word not in dictionary:
                wid = len(dictionary) + 1
                dictionary[word] = wid
            else:
                wid = dictionary[word]
            encoded.append(wid)
        drug_encoded[pid] = encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return drug_encoded, dictionary


def encode_other_drug(patient_drug: dict, pids: np.ndarray, dictionary: dict):
    print('encoding valid/test drug text ...')
    drug_encoded = dict()
    for i, pid in enumerate(pids):
        if pid not in patient_drug:
            drug_encoded[pid] = [0]
            continue
        words = extract_word(patient_drug[pid])
        encoded = []
        for word in words:
            if word in dictionary:
                encoded.append(dictionary[word])
        if len(encoded) == 0:
            encoded.append(0)
        drug_encoded[pid] = encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return drug_encoded
