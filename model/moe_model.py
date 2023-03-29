import torch
import os
from torch import nn
import torch.nn.functional as F

from .transformers.modeling_outputs import TokenClassifierOutput
from .transformers.models.bert import BertLayer
from .transformers.models.bert.modeling_bert import  BertModel


class MNERFiveExpertsModel(nn.Module):
    def __init__(self, label_list, args):
        super(MNERFiveExpertsModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        # Set the length of the prompt to the length of the main image plus the length of the secondary image
        self.resnet_prompt_len = args.img_resnet_main_prompt_len + args.img_resnet_aux_prompt_len
        self.resnet_main_len = self.resnet_prompt_len
        self.resnet_aux_len = self.resnet_prompt_len
        self.tags_prompt_len = self.resnet_prompt_len
        self.caption_prompt_len = self.resnet_prompt_len
        self.random_prompt_len = self.resnet_prompt_len
        self.prompt_len = self.resnet_prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config
        self.n_layer = self.bert_config.num_hidden_layers
        self.n_head = self.bert_config.num_attention_heads
        self.n_embd = self.bert_config.hidden_size // self.bert_config.num_attention_heads

        self.prefix_tokens = torch.arange(self.random_prompt_len).long()
        self.encoder_conv_embedding = torch.nn.Embedding(self.random_prompt_len, self.bert_config.hidden_size)

        # Projection of different image representations as prompts with the same dimensions
        self.encoder_conv_resnet_main = nn.Sequential(
            nn.Linear(in_features=3840, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_resnet_aux = nn.Sequential(
            nn.Linear(in_features=3840, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_tags = nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_caption = nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_random = nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        # Determine the contribution of each image representation in each layer
        self.gates = nn.ModuleList([nn.Linear(16 * 768, 5) for i in range(12)])
        self.gates_projection = nn.ModuleList([nn.Linear(768, 768) for i in range(12)])


        self.num_labels = len(label_list)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.alpha = args.alpha

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                main_features=None, aux_features=None, tag_embeddings=None, caption_embeddings=None):

        past_key_values, sum_key_values = self.get_visual_prompt(main_features, aux_features,  tag_embeddings, caption_embeddings)

        prompt_guids_length = past_key_values[0][0].shape[3]

        # attention_mask: bsz, seq_len
        # prompt attention， attention mask
        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)


        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                token_type_ids=token_type_ids,
                                past_key_values=past_key_values,
                                sum_key_values=sum_key_values,
                                gates=self.gates,
                                gates_projection=self.gates_projection,
                                return_dict=False)
        sequence_output = bert_output[0]  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)  # bsz, len, labels

        logits = emissions
        loss = None
        moe_loss = None
        total_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            moe_loss = bert_output[-1]
            total_loss = (1 - self.alpha) * loss + self.alpha * moe_loss
        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits
        )


    def get_visual_prompt(self, main_resnet, aux_resnet, tag_embeddings=None, caption_embeddings=None):
        bsz = main_resnet.size(0)


        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(self.bert.device)
        prefix_tokens = self.encoder_conv_embedding(prefix_tokens)

        past_key_values_main_resnet = self.encoder_conv_resnet_main(main_resnet)
        past_key_values_main_resnet = past_key_values_main_resnet.view(
            bsz,
            self.resnet_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_aux_resnet = self.encoder_conv_resnet_aux(aux_resnet)
        past_key_values_aux_resnet = past_key_values_aux_resnet.view(
            bsz,
            self.resnet_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_tags = self.encoder_conv_tags(tag_embeddings)
        past_key_values_tags = past_key_values_tags.view(
            bsz,
            self.tags_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_caption = self.encoder_conv_caption(caption_embeddings)
        past_key_values_caption = past_key_values_caption.view(
            bsz,
            self.caption_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_random = self.encoder_conv_random(prefix_tokens)
        past_key_values_random = past_key_values_random.view(
            bsz,
            self.random_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        total_key_values = torch.stack([past_key_values_main_resnet, past_key_values_aux_resnet, past_key_values_tags, past_key_values_caption,
                                      past_key_values_random])
        sum_key_values = total_key_values.sum(0)
        total_key_values = total_key_values.permute([3, 0, 1, 4, 2, 5]).split(2)
        sum_key_values = sum_key_values.permute([2, 0, 3, 1, 4]).split(2)


        return total_key_values, sum_key_values
    
    
class MREFiveModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(MREFiveModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.resnet_prompt_len = args.img_resnet_prompt_len + args.img_aux_resnet_prompt_len
        self.resnet_prompt_main_len = self.resnet_prompt_len
        self.resnet_prompt_aux_len = self.resnet_prompt_len
        self.tags_prompt_len = self.resnet_prompt_len
        self.caption_prompt_len = self.resnet_prompt_len
        self.random_prompt_len = self.resnet_prompt_len
        self.prompt_len = self.resnet_prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.bert_config = self.bert.config
        self.n_layer = self.bert_config.num_hidden_layers
        self.n_head = self.bert_config.num_attention_heads
        self.n_embd = self.bert_config.hidden_size // self.bert_config.num_attention_heads

        self.prefix_tokens = torch.arange(self.random_prompt_len).long()
        self.encoder_conv_embedding = torch.nn.Embedding(self.random_prompt_len, self.bert_config.hidden_size)

        self.encoder_conv_resnet_main = nn.Sequential(
            nn.Linear(in_features=3840, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_resnet_aux = nn.Sequential(
            nn.Linear(in_features=3840, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_tags = nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_caption = nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.encoder_conv_random = nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=self.n_layer * 2 * self.bert_config.hidden_size)
        )
        self.gates = nn.ModuleList([nn.Linear(16 * 768, 5) for i in range(12)])
        self.gates_projection = nn.ModuleList([nn.Linear(768, 768) for i in range(12)])

        self.fc = nn.Linear(self.bert.config.hidden_size * 2, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.leaky_rulu = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.alpha = args.alpha

        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                resnet_features_main=None, resnet_features_aux=None, tag_embeddings=None, caption_embeddings=None):

        past_key_values, sum_key_values = self.get_visual_prompt(resnet_features_main, resnet_features_aux, tag_embeddings, caption_embeddings)

        prompt_guids_length = past_key_values[0][0].shape[3]

        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                token_type_ids=token_type_ids,
                                past_key_values=past_key_values,
                                sum_key_values=sum_key_values,
                                gates=self.gates,
                                gates_projection=self.gates_projection,
                                return_dict=False)
        sequence_output = bert_output[0]  # bsz, len, hidden
        bsz, seq_len, hidden_size = sequence_output.shape
        entity_hidden_state = torch.Tensor(bsz, 2 * hidden_size)  # batch, 2*hidden

        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = sequence_output[i, head_idx, :].squeeze()
            tail_hidden = sequence_output[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.fc(entity_hidden_state)

        loss = None
        moe_loss = None
        total_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
            moe_loss = bert_output[-1]
            total_loss = (1 - self.alpha) * loss + self.alpha * moe_loss
            # loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits
        )

    def get_visual_prompt(self, resnet_features_main=None, resnet_features_aux=None, tag_embeddings=None, caption_embeddings=None):
        bsz = resnet_features_main.size(0)
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(self.bert.device)
        prefix_tokens = self.encoder_conv_embedding(prefix_tokens)

        past_key_values_resnet_main = self.encoder_conv_resnet_main(resnet_features_main)
        past_key_values_resnet_main = past_key_values_resnet_main.view(
            bsz,
            self.resnet_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_resnet_aux = self.encoder_conv_resnet_aux(resnet_features_aux)
        past_key_values_resnet_aux = past_key_values_resnet_aux.view(
            bsz,
            self.resnet_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_tags = self.encoder_conv_tags(tag_embeddings)
        past_key_values_tags = past_key_values_tags.view(
            bsz,
            self.tags_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_caption = self.encoder_conv_caption(caption_embeddings)
        past_key_values_caption = past_key_values_caption.view(
            bsz,
            self.caption_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_random = self.encoder_conv_random(prefix_tokens)
        past_key_values_random = past_key_values_random.view(
            bsz,
            self.random_prompt_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        total_key_values = torch.stack([past_key_values_resnet_main, past_key_values_resnet_aux, past_key_values_tags, past_key_values_caption,
                                        past_key_values_random])
        sum_key_values = total_key_values.sum(0)
        total_key_values = total_key_values.permute([3, 0, 1, 4, 2, 5]).split(2)
        sum_key_values = sum_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return total_key_values, sum_key_values


