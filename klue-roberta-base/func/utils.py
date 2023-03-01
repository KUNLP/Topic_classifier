import requests
import re, json, os
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset


def json2list(data_dir, file):
    json_list = []
    with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)
        for idx, contents in enumerate(input_data):
            content = {}
            content['idx'] = idx
            content['title'] = contents['title']
            content['predefined_news_category'] = contents['predefined_news_category']
            content['label'] = contents['label']
            content['url'] = contents['url']

            # content = []
            # content.append(idx)
            # content.append(contents['title'])
            # content.append(contents['predefined_news_category'])
            # content.append(contents['label'])
            # content.append(contents['url'])
            json_list.append(content)
    return json_list

json_list = json2list('../../data', 'ynat-v1_train.json')

def convert_data2dataset(datas, tokenizer, max_length):
    total_input_ids, total_attention_mask, total_labels = [], [], []
    total_idx = []
    for index, data in enumerate(tqdm(datas, desc='convert-data2dataset')):
        idx = data['idx']
        title = data['title']
        predefined_news_category = data['predefined_news_category']
        label = data['label']
        url = data['url']

        tokens_title = tokenizer.tokenize(title)
        input_ids = tokenizer(title)['input_ids']
        attention_mask = [1] * len(input_ids)
        padding = [1] * (max_length - len(input_ids))
        assert len(input_ids) == len(attention_mask)

        input_ids += padding
        attention_mask += padding

        total_idx.append(idx)
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_labels.append(label)

        if (index<2):
            print("*** Example ***")
            print("title : {}".format(title))
            print("tokens_title: {}".format(" ".join([str(x) for x in tokens_title])))
            print("input_ids: {}".format(" ".join([str(x) for x in total_input_ids[-1]])))
            print("attention_mask: {}".format(" ".join([str(x) for x in total_attention_mask[-1]])))
            print("label: {}".format(total_labels[-1]))
            print()

    total_idx = torch.tensor(total_idx, dtype=torch.long)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_labels = torch.tensor(total_labels, dtype=torch.long)

    dataset = TensorDataset(total_idx, total_input_ids, total_attention_mask, total_labels)
    return dataset

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
dataset = convert_data2dataset(json_list, tokenizer, 128)
