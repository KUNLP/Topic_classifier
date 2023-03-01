from original.func.main_func import train
import os

output_dir = '../../model'

config = {
    "mode": 'train',
    "epoch": 10,
    "learning_rate": 1e-5,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,
    "batch_size": 32,
    "max_length": 128,
    "num_labels": 7,
    "gradient_accumulation_steps": 1,
    "model_dir_path": output_dir,
}


if config['mode'] == 'train':
    train(config)
