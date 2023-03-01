from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import BertConfig, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from original.func.util_data import get_score, convert_data2dataset, make_json_list
import os
import numpy as np
from original.model.model import BertForSequenceClassification


def do_train(config, bert_model, optimizer, scheduler, train_dataloader, epoch, global_step):
    #batch 단위 별 score 를 담을 리스트
    #scores = []

    class2label = {'경제': 0, '사회': 1, '생활문화': 2, '세계': 3, '스포츠': 4, '정치': 5, 'IT과학': 6}

    # batch 단위 별 loss를 담을 리스트
    losses = []
    # 모델의 출력 결과와 실제 정답값을 담을 리스트
    total_predicts, total_corrects = [], []
    for step, batch in enumerate(tqdm(train_dataloader, desc="do_train(epoch_{})".format(epoch))):

        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
        idx = batch[4]
        idx = idx.cpu().detach().numpy().tolist()
        # 입력 데이터에 대한 출력과 loss 생성
        loss, predicts = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

#        cls_score, _ = predicts.max(dim=-1)
#        cls_score = cls_score.tolist()
#        scores += cls_score
        #cls_score, _ = predicts.max(dim=-1)
        #cls_score = cls_score.tolist()




        predicts = predicts.argmax(dim=-1)
        predicts = predicts.cpu().detach().numpy().tolist()
        labels = labels.detach().cpu().tolist()

        total_predicts += predicts
        total_corrects += labels

        if config["gradient_accumulation_steps"] > 1:
            loss = loss / config["gradient_accumulation_steps"]

        # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
        loss.backward()
        losses.append(loss.data.item())

        if (step + 1) % config["gradient_accumulation_steps"] == 0 or \
                (len(train_dataloader) <= config["gradient_accumulation_steps"] and (step + 1) == len(
                    train_dataloader)):
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), config["max_grad_norm"])

            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()
            scheduler.step()

            # 변화도를 0으로 변경
            bert_model.zero_grad()
            global_step += 1
    train_macro_precision, train_macro_recall, train_macro_f1_score, \
    train_micro_precision, train_micro_recall, train_micro_f1_score \
        = get_score(total_predicts, total_corrects, class2label)
    print(
        "TRAIN MACRO PRECISION : {}, TRAIN MACRO RECALL : {}, TRAIN MACRO F1_SCORE : {} ".format(train_macro_precision,
                                                                                                 train_macro_recall,
                                                                                                 train_macro_f1_score))
    print(
        "TRAIN MICRO PRECISION : {}, TRAIN MICRO RECALL : {}, TRAIN MICRO F1_SCORE : {} ".format(train_micro_precision,
                                                                                                 train_micro_recall,
                                                                                                 train_micro_f1_score))
    # 정확도 계산
    accuracy = accuracy_score(total_corrects, total_predicts)

    return accuracy, np.mean(losses), global_step


def do_evaluate(bert_model, test_dataloader, mode):
    class2label = {'경제': 0, '사회':1, '생활문화':2, '세계':3, '스포츠':4, '정치':5, 'IT과학':6}

    # 모델의 입력, 출력, 실제 정답값을 담을 리스트
    total_input_ids, total_predicts, total_corrects = [], [], []
    scores = []
    for step, batch in enumerate(tqdm(test_dataloader, desc="do_evaluate")):

        batch = tuple(t.cuda() for t in batch)
        input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]

        # 입력 데이터에 대한 출력 결과 생성
        predicts = bert_model(input_ids, attention_mask, token_type_ids)

        cls_score = predicts.max(dim=-1).values
        cls_score = cls_score.tolist()
        scores += cls_score

        predicts = predicts.argmax(dim=-1)
        predicts = predicts.cpu().detach().numpy().tolist()
        labels = labels.cpu().detach().numpy().tolist()
        input_ids = input_ids.cpu().detach().numpy().tolist()

        total_predicts += predicts
        total_corrects += labels
        total_input_ids += input_ids


    # 정확도 계산
    accuracy = accuracy_score(total_corrects, total_predicts)

    if(mode == "train"):
        valid_macro_precision, valid_macro_recall, valid_macro_f1_score, \
        valid_micro_precision, valid_micro_recall, valid_micro_f1_score = get_score(total_predicts, total_corrects,
                                                                                    class2label)
        print("VALID MACRO PRECISION : {}, VALID MACRO RECALL : {}, VALID MACRO F1_SCORE : {} ".format(
            valid_macro_precision, valid_macro_recall, valid_macro_f1_score))
        print("VALID MICRO PRECISION : {}, VALID MICRO RECALL : {}, VALID MICRO F1_SCORE : {} ".format(
            valid_micro_precision, valid_micro_recall, valid_micro_f1_score))
        return accuracy, scores
    else:
        return accuracy, total_input_ids, total_predicts, total_corrects


def train(config):
    bert_config = BertConfig.from_pretrained("bert-base-multilingual-cased", num_labels=config['num_labels'])
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", config=bert_config)

    bert_model.cuda()

    #train_datas = datas_list('../../../kpm_data/my_train.csv')
    train_datas = make_json_list('../../data', 'ynat-v1_train.json')
    train_dataset = convert_data2dataset(datas=train_datas, tokenizer=bert_tokenizer, max_length=config['max_length'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config['batch_size'])

    dev_datas = make_json_list('../../data', 'ynat-v1_dev_sample_10.json')
    dev_dataset = convert_data2dataset(datas=dev_datas, tokenizer=bert_tokenizer, max_length=config['max_length'])
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=config['batch_size'])

    t_total = len(train_dataloader) // config["gradient_accumulation_steps"] * config["epoch"]
    optimizer = AdamW(bert_model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=config["warmup_steps"],
                                                num_training_steps=t_total)

#    if os.path.isfile(os.path.join(config["model_dir_path"], "optimizer.pt")) and os.path.isfile(
#            os.path.join(config["model_dir_path"], "scheduler.pt")):
        # 기존에 학습했던 optimizer와 scheduler의 정보 불러옴
#        optimizer.load_state_dict(torch.load(os.path.join(config["model_dir_path"], "optimizer.pt")))
#        scheduler.load_state_dict(torch.load(os.path.join(config["model_dir_path"], "scheduler.pt")))

    global_step = 0
    bert_model.zero_grad()
    # 정확도 저장 변수
    max_test_accuracy = 0
    for epoch in range(config["epoch"]):
        bert_model.train()

        # 학습 데이터에 대한 정확도와 평균 loss
        train_accuracy, average_loss, global_step = do_train(config=config, bert_model=bert_model,
                                                             optimizer=optimizer, scheduler=scheduler,
                                                             train_dataloader=train_dataloader,
                                                             epoch=epoch + 1, global_step=global_step)

        print("train_accuracy : {}\taverage_loss : {}\n".format(round(train_accuracy, 4), round(average_loss, 4)))

        bert_model.eval()

        # 평가 데이터에 대한 정확도
        test_accuracy, scores = do_evaluate(bert_model=bert_model, test_dataloader=dev_dataloader, mode=config["mode"])

        print("test_accuracy : {}\n".format(round(test_accuracy, 4)))

        # 현재의 정확도가 기존 정확도보다 높은 경우 모델 파일 저장
        if (max_test_accuracy < test_accuracy):
            max_test_accuracy = test_accuracy

            output = os.path.join('../../model', "checkpoint-{}".format(epoch))

            if not os.path.exists(output):
                os.makedirs(output)

            bert_config.save_pretrained(output)
            bert_tokenizer.save_pretrained(output)
            bert_model.save_pretrained(output)