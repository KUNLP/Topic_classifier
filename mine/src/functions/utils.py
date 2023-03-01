import logging
import os, json
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


#필요한건 label이랑 title
#label이 골든임
#train 45678개
def make_train_json_list(args):
    json_list = []
    with open(os.path.join(args.data_dir, args.train_file), 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)
        for idx, content in enumerate(input_data):
            contents = []
            contents.append(idx)
            contents.append(content['title'])
            contents.append(content['label'])
            json_list.append(contents)
    return json_list


def make_dev_json_list(args):
    json_list = []
    with open(os.path.join(args.data_dir, args.predict_file), 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)
        for idx, content in enumerate(input_data):
            contents = []
            contents.append(idx)
            contents.append(content['title'])
            contents.append(content['label'])
            json_list.append(contents)
    return json_list



def labels2num(label):
    '스포츠', '세계', '정치', '사회', '생활문화', 'IT과학','경제'
    label2num = {'경제': 0, '사회':1, '생활문화':2, '세계':3, '스포츠':4, '정치':5, 'IT과학':6}
    return label2num[label]


def convert_datas2dataset(datas, args, tokenizer):
    total_idx, total_input_ids, total_attention_mask, total_token_type_ids, total_labels = [], [], [], [], []

    for index, data in enumerate(tqdm(datas, desc='convert_datas2dataset')):
        idx, title, label = data
        token_title = tokenizer.tokenize(title)

        tokens = ['[CLS]']
        tokens.extend(token_title)


        input_ids = [tokenizer._convert_token_to_id(token) for token in tokens]
        attention_mask = [1] * len(input_ids)
        #token_type_ids = tokenizer.create_token_type_ids_from_sequences(input_ids)
        assert len(input_ids) == len(attention_mask)
        padding = [0]*(args.max_seq_length - len(input_ids))

        input_ids.extend(padding)
        attention_mask.extend(padding)
        #token_type_ids.extend(padding)

        label = labels2num(label)

        total_idx.append(idx)
        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        #total_token_type_ids.append(token_type_ids)
        total_labels.append(label)

        if (index<2):
            print("*** example ***")
            print("title : {}".format(title))
            print("tokens : {}".format(tokens))
            print("input_ids : {}".format(input_ids))
            print("attention mask : {}".format(attention_mask))
            #print("token_type_ids : {}".format(token_type_ids))
            print("labels : {}".format(label))

    total_idx = torch.tensor(total_idx, dtype=torch.long)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    #total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_labels = torch.tensor(total_labels, dtype=torch.long)

    dataset = TensorDataset(total_idx, total_input_ids, total_attention_mask, total_labels)
    return dataset








