import argparse
import os
import logging
from attrdict import AttrDict
from transformers import BertTokenizer, BertConfig
from mine.src.model.model import BertForSequenceClassification
from mine.src.model.main_functions import train
from mine.src.functions.utils import init_logger


def create_model(args):
    config = BertConfig.from_pretrained(
        #'bert-base-uncased',
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "check={}".format(args.checkpoint)),
        num_labels=args.num_labels
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, "check={}".format(args.checkpoint)),
        do_lower_case=args.do_lower_case
    )

    model = BertForSequenceClassification.from_pretrained(
        #'bert-base-uncased',
        args.model_name_or_path if args.from_init_weight else os.path.join(args.output_dir, 'check-{}'.format(args.checkpoint)),
        config=config
    )
    model.to(args.device)
    return model, tokenizer


def main(cli_args):
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    logger = logging.getLogger(__name__)

    init_logger()

    model, tokenizer = create_model(args)

    if args.do_train:
        train(args, model, tokenizer, logger)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument('--data_dir', type=str, default='./data')
    cli_parser.add_argument('--output_dir', type=str, default='./model')
    cli_parser.add_argument('--model_name_or_path', type=str, default='./init_weight_bert')

    cli_parser.add_argument('--train_file', type=str, default='ynat-v1_train.json')
    cli_parser.add_argument('--predict_file', type=str, default='ynat-v1_dev_sample_10.json')

    # model hyper parameter
    cli_parser.add_argument('--max_seq_length', type=int, default=128)
    cli_parser.add_argument('--num_labels', type=int, default=7)
    cli_parser.add_argument('--vocab_size', type=int, default=32200)

    # training parameter
    cli_parser.add_argument('--learning_rate', type=float, default=1e-5)
    cli_parser.add_argument('--train_batch_size', type=int, default=32)
    cli_parser.add_argument('--eval_batch_size', type=int, default=32)
    cli_parser.add_argument('--num_train_epochs', type=int, default=5)
    cli_parser.add_argument('--threads', type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument('--max_grad_norm', type=int, default=1.0)
    cli_parser.add_argument("--do_lower_case", type=bool, default=False)
    cli_parser.add_argument("--logging_steps", type=int, default=1)

    #running mode
    cli_parser.add_argument("--from_init_weight", type=bool, default=True)
    cli_parser.add_argument("--do_train", type=bool, default=True)
    cli_parser.add_argument("--do_eval", type=bool, default=False)
    cli_parser.add_argument("--do_predict", type=bool, default=False)

    cli_args = cli_parser.parse_args()

    main(cli_args)