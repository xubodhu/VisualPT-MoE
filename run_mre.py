import os
import argparse
import logging
import sys

sys.path.append("..")
import datetime

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from modules.dataset import MREProcessor, MREFiveDataset
from model.moe_model import MREFiveModel
from modules.trainer import RETrainer


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=1234):
    """set random seed"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='./data/MNRE')
    parser.add_argument('--img_path',
                        type=str,
                        default='./data/MNRE_images')
    parser.add_argument('--img_resnet_features',
                        type=str,
                        default='./ner_img/resnet50_features_main_mre.pt')
    parser.add_argument("--img_aux_resnet_features_train",
                        type=str,
                        default='./ner_img/resnet50_features_aux_mre_train.pt')
    parser.add_argument("--img_aux_resnet_features_dev",
                        type=str,
                        default='./ner_img/resnet50_features_aux_mre_dev.pt')
    parser.add_argument("--img_aux_resnet_features_test",
                        type=str,
                        default='./ner_img/resnet50_features_aux_mre_test.pt')
    parser.add_argument('--random_prompt_len',
                        default=16,
                        type=int)
    parser.add_argument('--img_resnet_prompt_len',
                        default=4,
                        type=int,
                        help="prompt length")
    parser.add_argument('--img_aux_resnet_prompt_len',
                        default=12,
                        type=int,
                        help="prompt length")
    parser.add_argument('--tags_len',
                        default=16,
                        type=int)
    parser.add_argument('--caption_length',
                        type=int,
                        default=24)
    parser.add_argument('--img_object_tags',
                        type=str,
                        default='./ner_img/object_tags_mre.pt')
    parser.add_argument('--max_tags',
                        type=int,
                        default=3)
    parser.add_argument('--img_caption',
                        type=str,
                        default='./ner_img/caption_mre.pt')
    parser.add_argument("--alpha",
                        type=float,
                        default=0.00005)
    parser.add_argument('--dropout_prompt',
                        type=float,
                        default=0.3)
    parser.add_argument('--bert_name',
                        type=str,
                        default='../pretrained_models/bert-base-uncased')
    parser.add_argument('--num_epochs',
                        default=5,
                        type=int,
                        help="num training epochs")
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help="cuda or cpu")
    parser.add_argument('--batch_size',
                        default=8,
                        type=int,
                        help="batch size")
    parser.add_argument('--lr',
                        default=3e-5,
                        type=float,
                        help="learning rate")
    parser.add_argument('--prompt_lr',
                        default=5e-5,
                        type=float,
                        help="learning rate")
    parser.add_argument('--final_lr',
                        default=5e-3,
                        type=float,
                        help="learning rate")
    parser.add_argument('--warmup_ratio',
                        default=0.01,
                        type=float)
    parser.add_argument('--seed',
                        default=1234,
                        type=int,
                        help="random seed, default is 1")
    parser.add_argument('--total_prompt_len',
                        default=49,
                        type=int,
                        help="total prompt length")
    parser.add_argument('--prompt_dim',
                        default=1024,
                        type=int,
                        help="mid dimension of prompt project layer")
    parser.add_argument('--max_seq_length',
                        default=128,
                        type=int)
    parser.add_argument('--ignore_idx',
                        default=0,
                        type=int)
    parser.add_argument('--use_prompt',
                        default='true',
                        action='store_true')
    parser.add_argument('--sample_ratio',
                        default='1.0',
                        type=float)

    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    for k, v in vars(args).items():
        logger.info(" " + str(k) + " = %s", str(v))

    set_seed(args.seed)


    processor = MREProcessor(args)
    train_dataset = MREFiveDataset(processor, transform, args, 'ours_train.txt',
                                                          sample_ratio=args.sample_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_dataset = MREFiveDataset(processor, transform, args, 'ours_val.txt')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dataset = MREFiveDataset(processor, transform, args, 'ours_test.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


    re_dict = processor.get_relation_dict()
    num_labels = len(re_dict)
    tokenizer = processor.tokenizer


    model = MREFiveModel(num_labels, tokenizer, args)

    trainer = RETrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                         model=model, processor=processor, args=args, logger=logger)
    trainer.train()

