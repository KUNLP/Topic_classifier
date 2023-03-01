import os
import timeit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
import torch
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from mine.src.functions.utils import make_train_json_list, convert_datas2dataset, make_dev_json_list


def train(args, model, tokenizer, logger):
    train_json_list = make_train_json_list(args)
    train_dataset = convert_datas2dataset(train_json_list, args, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)



    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    tr_loss = 0.0
    global_step = 1
    model.zero_grad()
    mb = master_bar(range(args.num_train_epochs))


    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                'input_ids': batch[1],
                'attention_mask': batch[2],
                'labels': batch[3],
            }


            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step + 1), loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 반복 가능한 매개변수의 그래디언트 norm을 클립으로 자른다.
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # model save
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 모델 저장 디렉토리 생성
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 학습된 가중치 및 vocab 저장
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Validation Test!!
                    logger.info("***** Eval results *****")
                    results = evaluate(args, model, tokenizer, logger, global_step=global_step)

        mb.write("Epoch {} done".format(epoch + 1))
        return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, logger, global_step=""):
    dev_json_list = make_dev_json_list(args)
    dev_dataset = convert_datas2dataset(dev_json_list, args, tokenizer)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.train_batch_size)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(dev_dataset))
    # eval_batchsize 32
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in progress_bar(dev_dataloader):
        # 모델을 평가 모드로 변경
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[1],
                'attention_mask': batch[2],
            }

            outputs = model(**inputs)




