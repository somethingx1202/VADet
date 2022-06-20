# -*- coding: utf8 -*-

import torch
import torch.nn as nn

from transformers import AlbertConfig
from vad_unsup_models import TopicDrivenMaskedLM
from transformers import AlbertTokenizerFast

import sentencepiece as spm
import os, sys

# sys.path.insert(0, "../input/sentencepiece-pb2")
# Important: no '/' in the tail please!
sys.path.insert(0, "../datasets/sentencepiece-pb2")
import sentencepiece_pb2
# See google colab sentence piece
# https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb#scrollTo=SUcAbKnRVAv6
# See
# !pip install protobuf
# !wget https://raw.githubusercontent.com/google/sentencepiece/master/python/sentencepiece_pb2.py
# in the same page
# https://github.com/google/sentencepiece/tree/master/python/src/sentencepiece

# About the version problem Missing key(s) in state_dict: "transformer.embeddings.position_ids".,
# see https://github.com/huggingface/transformers/issues/6882

MODEL_PATHS = {
    'bert-base-uncased': '../datasets/bertconfigs/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/',
    'bert-large-uncased-whole-word-masking-finetuned-squad': '../datasets/bertconfigs/wwm_uncased_L-24_H-1024_A-16/wwm_uncased_L-24_H-1024_A-16/',
    'albert-large-v2': '../datasets/albertconfigs/albert-large-v2/albert-large-v2/',
    'vadlm-albert-large-v2': '../datasets/albertconfigs/vadlm-albert-large-v2/vadlm-albert-large-v2/',
    'albert-base-v2': '../datasets/albertconfigs/albert-base-v2/albert-base-v2/',
    'distilbert': '../datasets/albertconfigs/distilbert/distilbert/',
}

TRANSFORMERS = {
    'vadlm-albert-large-v2': (TopicDrivenMaskedLM, 'vadlm-albert-large-v2', AlbertConfig),
}


class VADTransformer(nn.Module):
    def __init__(self, model, nb_layers=1, nb_ft=None, nb_class=None, pretrained=False, nb_cate=None, multi_sample_dropout=False, training=False, use_squad_weights=True):
        super().__init__()
        self.name = model
        self.nb_layers = nb_layers
        self.multi_sample_dropout = multi_sample_dropout

        # padding token_id
        self.pad_idx = 1 if "roberta" in self.name else 0

        # So here actualy load model class and config from a dict. This is unlike C.
        model_class, _, config_class = TRANSFORMERS[model]

        if 'vadlm-albert-large-v2' in self.name:
            config = config_class.from_pretrained(MODEL_PATHS[model], output_hidden_states=True)
            self.config = config
            if pretrained:
                print('I am loading from pre-trained')
                self.transformer = model_class.from_pretrained('../datasets/albertconfigs/vadlm-albert-large-v2/vad-cache/', config=config)
            else:
                self.transformer = model_class(config)
        else:
            assert False


        if "distil" in self.name:
            self.nb_features = self.transformer.transformer.layer[-1].ffn.lin2.out_features
        elif "vadlm-albert-large-v2" in self.name:
            self.nb_features = self.transformer.albert.encoder.albert_layer_groups[-1].albert_layers[-1].ffn_output.out_features
        elif "albert" in self.name:
            # The last layers
            self.nb_features = self.transformer.encoder.albert_layer_groups[-1].albert_layers[-1].ffn_output.out_features
        else:
            self.nb_features = self.transformer.pooler.dense.out_features

        if nb_ft is None:
            # Number of features
            nb_ft = self.nb_features

        self.logits = nn.Sequential(
            nn.Linear(self.nb_features * self.nb_layers, nb_ft),
            nn.Tanh(),
            nn.Linear(nb_ft, nb_class),
        )

        self.cates = nn.Linear(self.nb_features * self.nb_layers, nb_cate)

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, tokens, token_type_ids, aspect_tokens, aspect_token_type_ids, lm_inputs=None, lm_labels=None):
        if "distil" in self.name:
            hidden_states = self.transformer(
                tokens,
                attention_mask=(tokens != self.pad_idx).long(),
            )[-1]
        else:
            hidden_states = self.transformer(
                input_ids=tokens,
                attention_mask=(tokens != self.pad_idx).long(),
                token_type_ids=token_type_ids,
            )[-1]
            if lm_labels is not None and lm_inputs is not None:
                lm_loss = self.transformer(
                    labels=lm_labels,
                    input_ids=lm_inputs,
                    attention_mask=(lm_inputs != self.pad_idx).long(),
                    token_type_ids=token_type_ids,
                )[0]
            else:
                lm_loss = 0

            hidden_states_apsect_span = self.transformer(
                input_ids=aspect_tokens,
                attention_mask=(aspect_tokens != self.pad_idx).long(),
                token_type_ids=aspect_token_type_ids,
            )[-1]


        halfway_hidden_states = hidden_states[:self.config.num_hidden_layers // 2]
        aggd_hidden_semantic_state = self.transformer.att(halfway_hidden_states[-1])
        mu, sigma_log_pow = self.transformer.vae_encoder(aggd_hidden_semantic_state)

        halfway_hidden_states_aspect_span = hidden_states_apsect_span[:self.config.num_hidden_layers // 2]
        aggd_hidden_semantic_state_aspect_span = self.transformer.att(halfway_hidden_states_aspect_span[-1])
        mu_asp, sigma_log_pow_asp = self.transformer.vae_encoder(aggd_hidden_semantic_state_aspect_span)

        exp_sum_sigma_log_pow_asp_rev = torch.exp(torch.mean(-sigma_log_pow_asp, dim=1))
        extended_exp_sum_sigma_log_pow_asp_rev = exp_sum_sigma_log_pow_asp_rev.unsqueeze(1)
        
        kld_loss = -0.5 * torch.mean(
            (1 + sigma_log_pow - sigma_log_pow_asp - extended_exp_sum_sigma_log_pow_asp_rev * ((mu - mu_asp)**2) -
             torch.exp(-sigma_log_pow_asp + sigma_log_pow)),
            dim=1)
        kld_loss = torch.mean(kld_loss, dim=0)

        kld_loss_asp = -0.5 * torch.mean(
            (1 + sigma_log_pow_asp - mu_asp**2 -
             torch.exp(sigma_log_pow_asp)),
            dim=1)
        kld_loss_asp = torch.mean(kld_loss_asp, dim=0)

        hidden_states = hidden_states[::-1]
        features = torch.cat(hidden_states[:self.nb_layers], -1)

        if self.multi_sample_dropout and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
            clss = torch.mean(
                torch.stack(
                    [self.cates(self.high_dropout(features[:,0,:])) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(features)
            clss = self.cates(self.high_dropout(features[:,0,:]))

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits, clss, kld_loss + kld_loss_asp, lm_loss


# ///////////////////// PerText Encoder Decoder
class EncodedText:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets

        
class SentencePieceTokenizer:
    def __init__(self, model_path, lowercase=True):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_path))
        self.lowercase = lowercase
    
    def encode(self, sentence):
        if self.lowercase:
            sentence = sentence.lower()
            
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            # The tokenizer has provided the begin and end tokens
            offsets.append((piece.begin, piece.end))
        return EncodedText(tokens, offsets)

    def decode_ids(self, ids_list):
        '''
        tokenize only used for testing, attributes to google sentence piece colab
        '''  
        decoded_from_ids = self.sp.decode_ids(ids_list)
        return decoded_from_ids


def create_tokenizer_and_tokens(df, config):
    '''
    Use albert-large-v2 tokenizer
    '''

    if 'albert-large-v2' in config.selected_model:
        # tokenizer = AlbertTokenizerFast.from_pretrained('albert-large-v2')
        # tokenizer.save_pretrained('../datasets/albertconfigs/albert-large-v2/tokenizer')
        tokenizer = AlbertTokenizerFast.from_pretrained('../datasets/albertconfigs/albert-large-v2/tokenizer')
        special_tokens = {
            "sep": tokenizer.sep_token_id,
            "cls": tokenizer.cls_token_id,
            "pad": tokenizer.pad_token_id,
        }
    elif "albert" in config.selected_model:
        # They used a stand_alone tokenizer in albert
        # tokenizer = SentencePieceTokenizer(f'{MODEL_PATHS[config.selected_model]}/{config.selected_model}-spiece.model')
        tokenizer = SentencePieceTokenizer(f'{MODEL_PATHS["albert-large-v2"]}/albert-large-v2-spiece.model')
        special_tokens = {
            'cls': 2,
            'sep': 3,
            'pad': 0,
        }
    else:
        assert False
        tokenizer = BertWordPieceTokenizer(
            MODEL_PATHS[config.selected_model] + 'vocab.txt',
            lowercase=config.lowercase,
#             add_special_tokens=False  # This doesn't work smh
        )

        special_tokens = {
            'cls': tokenizer.token_to_id('[CLS]'),
            'sep': tokenizer.token_to_id('[SEP]'),
            'pad': tokenizer.token_to_id('[PAD]'),
        }

    ids = {}
    offsets = {}

    if 'albert-large-v2' in config.selected_model:
        longest_input_ids_text_len_for_display = 0
        texts = df["clean_text"].unique()
        for text in texts:
            encoding = tokenizer(
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False)
            # assert text not in ids
            ids[text] = encoding["input_ids"]
            offsets[text] = encoding["offset_mapping"]
            if longest_input_ids_text_len_for_display < len(encoding["input_ids"]):
                longest_input_ids_text_len_for_display = len(encoding["input_ids"])
        print(f"longest_input_ids_text_len_for_display = {longest_input_ids_text_len_for_display}")

    elif "albert" in config.selected_model:
        longest_input_ids_text_len_for_display = 0
        texts = df["clean_text"].unique()
        for text in texts:
            tokenized = tokenizer.encode(text)
            ids[text] = tokenized.ids
            offsets[text] = tokenized.offsets
            if longest_input_ids_text_len_for_display < len(tokenized.ids):
                longest_input_ids_text_len_for_display = len(tokenized.ids)
        print(f"longest_input_ids_text_len_for_display = {longest_input_ids_text_len_for_display}")
    else:
        texts = df["clean_text"].unique()
        for text in texts:
            encoding = tokenizer(
                text,
                return_token_type_ids=True,
                return_offsets_mapping=True,
                return_attention_mask=False,
                add_special_tokens=False,
            )
            ids[text] = encoding["input_ids"]
            offsets[text] = encoding["offset_mapping"]
    precomputed_tokens_and_offsets = {"ids": ids, "offsets": offsets}

    return tokenizer, special_tokens, precomputed_tokens_and_offsets
