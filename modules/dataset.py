import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms


class MNERProcessor(object):
    def __init__(self, args) -> None:
        self.data_path = args.data_path
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_name, do_lower_case=True)

    def load_from_file(self, file_name, sample_ratio=1.0):
        file_name = os.path.join(self.data_path, file_name)
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
                                                                                    len(imgs))

        # sample data, only for low-resource
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words) * sample_ratio))
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(
                len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs}

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
        return label_mapping
    # Not used yet
    def get_auxlabel_mapping(self):
        LABEL_LIST = ["O", "B", "I"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
        return label_mapping


class MREProcessor(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.re_path = os.path.join(self.data_path, "ours_rel2id.json")
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})


    def load_from_file(self, file_name, sample_ratio=1.0):
        load_file = os.path.join(self.data_path, file_name)
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])  # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        # sample
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(words))), k=int(len(words) * sample_ratio))
            sample_words = [words[idx] for idx in sample_indexes]
            sample_relations = [relations[idx] for idx in sample_indexes]
            sample_heads = [heads[idx] for idx in sample_indexes]
            sample_tails = [tails[idx] for idx in sample_indexes]
            sample_imgids = [imgids[idx] for idx in sample_indexes]

            assert len(sample_words) == len(sample_relations) == len(sample_imgids), "{}, {}, {}".format(
                len(sample_words), len(sample_relations), len(sample_imgids))
            return {'words': sample_words, 'relations': sample_relations, 'heads': sample_heads, 'tails': sample_tails, \
                    'imgids': sample_imgids}

        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids, 'dataid': dataid}

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding='utf-8') as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict


class MNERFiveExpertsDataset(Dataset):
    def __init__(self, processor, transform, args, file_name, ignore_idx=0, sample_ratio=1.0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(file_name, sample_ratio=sample_ratio)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        # Maximum length of text
        self.max_seq_length = args.max_seq_length
        self.ignore_idx = ignore_idx
        self.device = args.device
        # Maximum number of objects per image 
        self.max_tags = args.max_tags
        # Load the features of the main image
        self.img_main_resnet_features = torch.load(args.img_main_resnet_features)
        # Load the features of the aux images
        self.img_aux_resnet_features = torch.load(args.img_aux_resnet_features)
        self.img_resnet_main_prompt_len = args.img_resnet_main_prompt_len
        self.img_resnet_aux_prompt_len = args.img_resnet_aux_prompt_len
        self.img_resnet_prompt_len = args.img_resnet_main_prompt_len + args.img_resnet_aux_prompt_len
        # Load image object features
        self.img_object_tags = torch.load(args.img_object_tags)
        # Unify the length of the prompt
        self.tags_length = self.img_resnet_prompt_len
        self.caption_length = self.img_resnet_prompt_len
        # Load image caption features
        self.img_caption = torch.load(args.img_caption)
        # Convert text (image objects and caption) to a vector
        self.bert_embedding = BertModel.from_pretrained(args.bert_name).get_input_embeddings().to(args.device)

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], \
                                     self.data_dict['imgs'][idx]
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(-100)
        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 2)]
            labels = labels[0:(self.max_seq_length - 2)]

        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq_length, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        labels = [-100] + labels + [-100] + [-100] * (
                self.max_seq_length - len(labels) - 2)



        main_image = self.img_main_resnet_features[img]
        image_len, image_dim = main_image.shape
        zeros = torch.zeros((self.img_resnet_prompt_len - image_len), image_dim)
        main_image = torch.cat([main_image, zeros])
        aux_images = self.img_aux_resnet_features[img]
        image_len, image_dim = aux_images.shape
        zeros = torch.zeros((self.img_resnet_prompt_len - image_len), image_dim)
        aux_images = torch.cat([aux_images, zeros])


        image_object_tags = self.img_object_tags[img][:self.max_tags]
        tokens = []
        for tag in image_object_tags:
            tok = self.tokenizer.tokenize(tag)
            tokens.extend(tok)
        tokens = tokens[:self.tags_length]
        tokens += ['[PAD]'] * (self.tags_length - len(tokens))
        tag_input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tag_input_ids = torch.tensor(tag_input_ids, dtype=torch.int64).to(self.device)
        tag_embeddings = self.bert_embedding(tag_input_ids).cpu()

        caption = self.img_caption[img]
        tokens = self.tokenizer.tokenize(caption)

        tokens += ['[PAD]'] * (self.caption_length - len(tokens))
        tokens = tokens[:self.caption_length]
        caption_input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        caption_input_ids = torch.tensor(caption_input_ids, dtype=torch.int64).to(self.device)
        caption_embeddings = self.bert_embedding(caption_input_ids).cpu()

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)

        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask), torch.tensor(labels), main_image, aux_images, tag_embeddings, caption_embeddings

class MREFiveDataset(Dataset):
    def __init__(self, processor, transform, args, file_name, sample_ratio=1.0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = self.processor.load_from_file(file_name, sample_ratio)
        self.re_dict = self.processor.get_relation_dict()
        self.device = args.device
        self.tokenizer = self.processor.tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_tags = args.max_tags
        self.img_resnet_features = torch.load(args.img_resnet_features)
        self.img_aux_resnet_features = None
        if "train" in file_name:
            self.img_aux_resnet_features = torch.load(args.img_aux_resnet_features_train)
        elif "val" in file_name:
            self.img_aux_resnet_features = torch.load(args.img_aux_resnet_features_dev)
        elif "test" in file_name:
            self.img_aux_resnet_features = torch.load(args.img_aux_resnet_features_test)

        self.img_resnet_prompt_len = args.img_resnet_prompt_len
        self.img_aux_resnet_prompt_len = args.img_aux_resnet_prompt_len
        self.img_object_tags = torch.load(args.img_object_tags)
        self.img_caption = torch.load(args.img_caption)
        self.tags_length = args.img_resnet_prompt_len + args.img_aux_resnet_prompt_len
        self.caption_length = args.img_resnet_prompt_len + args.img_aux_resnet_prompt_len
        self.bert_embedding = BertModel.from_pretrained(args.bert_name).get_input_embeddings().to(args.device)

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, img = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                   self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                   self.data_dict['imgids'][idx]
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']


        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq_length, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        re_label = self.re_dict[relation]  # label to id

        main_image = self.img_resnet_features[img]
        image_len, image_dim = main_image.shape
        zeros = torch.zeros((self.img_resnet_prompt_len + self.img_aux_resnet_prompt_len - image_len), image_dim)
        main_image = torch.cat([main_image, zeros])
        aux_images = self.img_aux_resnet_features[item_id]
        image_len, image_dim = aux_images.shape
        zeros = torch.zeros((self.img_resnet_prompt_len + self.img_aux_resnet_prompt_len - image_len), image_dim)
        aux_images = torch.cat([aux_images, zeros], dim=0)

        image_object_tags = self.img_object_tags[img][:self.max_tags]
        caption = self.img_caption[img]

        tokens = []
        for tag in image_object_tags:
            tok = self.tokenizer.tokenize(tag)
            tokens.extend(tok)
        tokens = tokens[:self.tags_length]
        tokens += ['[PAD]'] * (self.tags_length - len(tokens))
        tag_input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tag_input_ids = torch.tensor(tag_input_ids, dtype=torch.int64).to(self.device)
        tag_embeddings = self.bert_embedding(tag_input_ids).cpu()

        tokens = self.tokenizer.tokenize(caption)
        tokens = tokens[:self.caption_length]
        tokens += ['[PAD]'] * (self.caption_length - len(tokens))
        caption_input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        caption_input_ids = torch.tensor(caption_input_ids, dtype=torch.int64).to(self.device)
        caption_embeddings = self.bert_embedding(caption_input_ids).cpu()


        return input_ids, token_type_ids, attention_mask, torch.tensor(
            re_label), main_image, aux_images, tag_embeddings, caption_embeddings



