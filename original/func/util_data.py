from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
import numpy as np
import os , json

def make_json_list(data_dir, file):
    json_list = []
    with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)
        for idx, content in enumerate(input_data):
            contents = []
            contents.append(idx)
            contents.append(content['title'])
            contents.append(content['label'])
            json_list.append(contents)
    return json_list


def convert_data2dataset(datas, tokenizer, max_length):
    #datas : train_datas = datas_list -> 리스트 형식으로 데이터 입력
    total_input_ids, total_attention_mask, total_token_type_ids, total_labels = [], [], [], []
    total_idx = []

    for index, data in enumerate(tqdm(datas, desc="convert_data2dataset")):
        # id 안써도 되니까 _ 처리
        idx, title, label = data

        class2label = {'경제': 0, '사회': 1, '생활문화': 2, '세계': 3, '스포츠': 4, '정치': 5, 'IT과학': 6}
        label = class2label[label]

        tokens_ = tokenizer.tokenize(title)
        # token cls .... sep 형태로 만들어줌

        tokens = ["[CLS]"] + tokens_


        input_ids = [tokenizer._convert_token_to_id(token) for token in tokens]

        assert len(input_ids) <= max_length

        # 실제 길이만큼 1 mask 리스트 생성
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)


        padding = [0] * (max_length - len(input_ids))
        assert len(input_ids) == len(attention_mask) == len(token_type_ids)
        # 동일한 크기 리스트
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        total_input_ids.append(input_ids)
        total_attention_mask.append(attention_mask)
        total_token_type_ids.append(token_type_ids)
        total_labels.append(int(label))
        total_idx.append(int(idx))
        # label 은 인덱스값으로 되어있다.

        # 몇몇 샘플 변환된거 출력하는거임
        if (index < 2):
            print("*** Example ***")
            print("title : {}".format(title))
            print("tokens: {}".format(" ".join([str(x) for x in tokens])))
            print("input_ids: {}".format(" ".join([str(x) for x in total_input_ids[-1]])))
            print("attention_mask: {}".format(" ".join([str(x) for x in total_attention_mask[-1]])))
            print("token_type_ids: {}".format(" ".join([str(x) for x in total_token_type_ids[-1]])))
            print("label: {}".format(total_labels[-1]))
            print()

    total_idx = torch.tensor(total_idx, dtype=torch.long)
    total_input_ids = torch.tensor(total_input_ids, dtype=torch.long)
    total_attention_mask = torch.tensor(total_attention_mask, dtype=torch.long)
    total_token_type_ids = torch.tensor(total_token_type_ids, dtype=torch.long)
    total_labels = torch.tensor(total_labels, dtype=torch.long)

    dataset = TensorDataset(total_input_ids, total_attention_mask, total_token_type_ids, total_labels, total_idx)

    return dataset



def get_score(predicts, corrects, class2label):

    label2class = {0:'경제', 1:'사회', 2:'생활문화', 3:'세계',4:'스포츠',5:'정치',6:'IT과학'}
    predicts = [label2class[predict] for predict in predicts]
    corrects = [label2class[correct] for correct in corrects]
    result = {}


    def get_score_one_class(predicts, corrects, value):
        TP, FP, FN, TN = 0, 0, 0, 0
        for correct, predict in zip(corrects, predicts):
            if(correct == value and predict == value):
                TP += 1
            elif(correct != value and predict == value):
                FP += 1
            elif(correct == value and predict != value):
                FN += 1
            elif(correct != value and predict != value):
                TN += 1

        if(TP == 0):
            precision, recall, f1_score, accuracy = 0, 0, 0, 0
        else:
            precision = float(TP)/(TP+FP)
            recall = float(TP)/(TP+FN)
            f1_score = (2*precision*recall)/(precision+recall)
            accuracy = float(TP+TN)/(TP+FN+FP+TN)

        return precision, recall, f1_score, accuracy, TP, FP, FN, TN

    values = list(label2class.values())
    for value in values:
        precision, recall, f1_score, accuracy, TP, FP, FN, TN = get_score_one_class(predicts, corrects, value)
        result[value] = {"precision":precision, "recall":recall, "f1_score":f1_score, "accuracy":accuracy,
                         "TP":TP, "FP":FP, "FN":FN, "TN":TN}

    macro_precision = np.sum([result[value]["precision"] for value in values]) / len(values)
    macro_recall = np.sum([result[value]["recall"] for value in values]) / len(values)
    macro_f1_score = np.sum([result[value]["f1_score"] for value in values]) / len(values)

    total_TP = np.sum([result[value]["TP"] for value in values])
    total_FP = np.sum([result[value]["FP"] for value in values])
    total_FN = np.sum([result[value]["FN"] for value in values])
    total_TN = np.sum([result[value]["TN"] for value in values])

    if (total_TP == 0):
        micro_precision, micro_recall, micro_f1_score, accuracy = 0, 0, 0, 0
    else:
        micro_precision = float(total_TP) / (total_TP + total_FP)
        micro_recall = float(total_TP) / (total_TP + total_FN)
        micro_f1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    # for value in values:
    #     precision, recall, f1_score = result[value]["precision"], result[value]["recall"], result[value]["f1_score"]
    #     TP, FP, FN, TN = result[value]["TP"], result[value]["FP"], result[value]["FN"], result[value]["TN"]
    #
    #     print("Precision from {} : ".format(value) + str(round(precision, 4)))
    #     print("Recall from {} : ".format(value) + str(round(recall, 4)))
    #     print("F1_score from {} : ".format(value) + str(round(f1_score, 4)))
    #
    #     print("True Positive from {} : ".format(value) + str(TP))
    #     print("False Positive from {} : ".format(value) + str(FP))
    #     print("False Negative from {} : ".format(value) + str(FN))
    #     print("True Negative from {} : ".format(value) + str(TN) + "\n")

    return round(macro_precision, 4), round(macro_recall, 4), round(macro_f1_score, 4), \
           round(micro_precision, 4), round(micro_recall, 4), round(micro_f1_score, 4)
