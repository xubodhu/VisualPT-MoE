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
from modules.dataset import MNERProcessor, MNERFiveExpertsDataset
from model.moe_model import MNERFiveExpertsModel
from modules.trainer import NERTrainer

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


logging.basicConfig(format = '%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
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
                        default='./data/twitter2017')
    parser.add_argument('--img_path',
                        type=str,
                        default='./data/twitter2017_images')
    parser.add_argument('--img_resnet_features',
                        type=str,
                        default='./ner_img/resnet50_features_2017.pt')
    parser.add_argument('--img_main_resnet_features',
                        type=str,
                        default='./ner_img/resnet50_features_main_2017.pt')
    parser.add_argument('--img_aux_resnet_features',
                        type=str,
                        default='./ner_img/resnet50_features_aux_2017.pt')
    parser.add_argument("--img_resnet_main_prompt_len",
                        default=4,
                        type=int)
    parser.add_argument("--img_resnet_aux_prompt_len",
                        default=12,
                        type=int)
    parser.add_argument('--random_prompt_len',
                        default=16,
                        type=int)
    parser.add_argument('--img_resnet_prompt_len',
                        default=49,
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
                        default='./ner_img/object_tags_2017.pt')
    parser.add_argument('--max_tags',
                        type=int,
                        default=3)
    parser.add_argument('--img_caption',
                        type=str,
                        default='./ner_img/caption_2017.pt')
    parser.add_argument('--alpha',
                        type=float,
                       default=0.0001)
    parser.add_argument('--dropout_prompt',
                        type=float,
                        default=0.3)
    parser.add_argument('--bert_name',
                        type=str,
                        default='../pretrained_models/bert-base-uncased')
    parser.add_argument('--num_epochs',
                        default=4,
                        type=int,
                        help="num training epochs")
    parser.add_argument('--device',
                        default='cuda',
                        type=str,
                        help="cuda or cpu")
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help="batch size")
    parser.add_argument('--lr',
                        default=5e-5,
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

    for k,v in vars(args).items():
        logger.info(" " + str(k) +" = %s", str(v))

    set_seed(args.seed)
    
    processor = MNERProcessor(args)
    train_dataset = MNERFiveExpertsDataset(processor, transform, args, 'train.txt', sample_ratio=args.sample_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_dataset = MNERFiveExpertsDataset(processor, transform, args, 'valid.txt')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dataset = MNERFiveExpertsDataset(processor, transform, args, 'test.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    label_mapping = processor.get_label_mapping()
    label_list = list(label_mapping.keys())
    model = MNERFiveExpertsModel(label_list, args)
    trainer = NERTrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                      model=model, label_map=label_mapping, args=args, logger=logger)
    trainer.train()

