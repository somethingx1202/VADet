import re
import os
import gc
import pickle
import random
from selectors import EpollSelector
# from typing_extensions import assert_type
import sklearn
import warnings
import datetime
import numpy as np
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
import seaborn as sns
import matplotlib.pyplot as plt

# from tokenizers import *
from datetime import date
# from transformers import *

from transformers import set_seed
# , AdamW
from torch.optim import AdamW

from tqdm import trange, tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold
from torch.utils.data import SequentialSampler, RandomSampler
# WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import csv
import copy

from sklearn.metrics import f1_score

from model_config_tokenizer import VADTransformer, create_tokenizer_and_tokens

SEED = 42
K = 5

DATA_PATH = '../datasets/Vaccine-Attitude-Dataset/'


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_weights(model, filename, verbose=1, cp_folder=""):
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    
    try:
        # Seems that strict wasn't defined if run in standalone
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        # But the keys in the pikle doesn't match the latest transformer version
        print('Loading weights to cpu using torch.load instead of strict')
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )
    return model


def trim_tensors(tokens, input_ids, lm_input_ids, lm_labels, model_name='bert', min_len=10):
    pad_token = 1 if "roberta" in model_name else 0
    # This finds the longest sequence len, except the padded token.
    max_len = max(torch.max(torch.sum((tokens != pad_token), 1)), min_len)
    return tokens[:, :max_len], input_ids[:, :max_len], lm_input_ids[:, :max_len], lm_labels[:, :max_len]


import os
import sys


# ///////////////////// Deal with dataframes with pandas to prepare the training list
def process_text(text):
    text = re.sub('I-year', '1-year', text)
    text = re.sub('-OR-', " or ", text)
    text = re.sub('-', ' ', text)
    return text


def clean_spaces4text(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)
    txt = re.sub('\s+$', '', txt)
    # if txt.find('  ') != -1:
    #     print('Error: double space')
    return txt


def clean_spaces(txt):
    txt = re.sub('\n', ' ', txt)
    txt = re.sub('\t', ' ', txt)
    txt = re.sub('\r', ' ', txt)

    # txt = txt.replace('\u2581',' ')
    # txt = txt.replace('\u0020',' ')
    # txt = txt.replace('\u00A0',' ')
    # txt = txt.replace('\u1680',' ')
    # txt = txt.replace('\u180E',' ')
    # txt = txt.replace('\u2000',' ')
    # txt = txt.replace('\u2001',' ')
    # txt = txt.replace('\u2002',' ')
    # txt = txt.replace('\u2003',' ')
    # txt = txt.replace('\u2004',' ')
    # txt = txt.replace('\u2005',' ')
    # txt = txt.replace('\u2006',' ')
    # txt = txt.replace('\u2007',' ')
    # txt = txt.replace('\u2008',' ')
    # txt = txt.replace('\u2009',' ')
    # txt = txt.replace('\u200A',' ')
    # txt = txt.replace('\u200B',' ')
    # txt = txt.replace('\u202F',' ')
    # txt = txt.replace('\u205F',' ')
    # txt = txt.replace('\u3000',' ')
    # txt = txt.replace('\uFEFF',' ')
    # txt = txt.replace('\u2423',' ')
    # txt = txt.replace('\u2422',' ')
    # txt = txt.replace('\u2420',' ')

    # looks that the irregular character appears there
    # txt = re.sub('[\u0080-\u0090]', '_', txt)

    # txt = re.sub('\u0091', '\'', txt)
    # txt = re.sub('\u0092', '\'', txt)
    # txt = re.sub('\u0093', '\"', txt)
    # txt = re.sub('\u0094', '\"', txt)
    # txt = re.sub('\u0095', '.', txt)
    # txt = re.sub('\u0096', '-', txt)
    # txt = re.sub('\u0097', '-', txt)
    # txt = re.sub('\u0098', '~', txt)
    # txt = re.sub('\u0099', '#', txt)
    # txt = re.sub('\u009a', 'S', txt)
    # txt = re.sub('\u009b', '>', txt)
    # txt = re.sub('\u009c', '.', txt)
    # txt = re.sub('\u009d', '_', txt)
    # txt = re.sub('\u009e', '_', txt)
    # txt = re.sub('\u009f', '_', txt)

    # txt = re.sub('[\u00A0-\u00BF]', '_', txt)
    # txt = re.sub('[\u00A0-\u00A7]', '_', txt)
    # txt = re.sub('[\u00A8-\u00AF]', '_', txt)
    # txt = re.sub('[\u00B0-\u00B7]', '_', txt)
    txt = re.sub('\u00B0', ' ', txt) # degree sign
    
    # txt = re.sub('\u00C0|\u00C1|\u00C2|\u00C3|\u00C4|\u00C5|\u00C6', 'A', txt)
    # txt = re.sub('\u00C7', 'C', txt)
    # txt = re.sub('\u00C8|\u00C9|\u00CA|\u00CB', 'E', txt)
    # txt = re.sub('\u00CC|\u00CD|\u00CE|\u00CF', 'I', txt)
    # txt = re.sub('\u00D0', 'T', txt)
    # txt = re.sub('\u00D1', 'N', txt)
    # txt = re.sub('\u00D2|\u00D3|\u00D4|\u00D5|\u00D6', 'O', txt)
    # txt = re.sub('\u00D7', '*', txt)
    # txt = re.sub('\u00D8', '0', txt)
    # txt = re.sub('\u00D9|\u00DA|\u00DB|\u00DC', 'U', txt)
    # txt = re.sub('\u00DD', 'Y', txt)
    # txt = re.sub('\u00DE', 'S', txt)
    # txt = re.sub('\u00DF', 's', txt)
    # txt = re.sub('\u00E0|\u00E1|\u00E2|\u00E3|\u00E4|\u00E5|\u00E6', 'a', txt)
    # txt = re.sub('\u00E7', 'c', txt)
    # txt = re.sub('\u00E8|\u00E9|\u00EA|\u00EB', 'e', txt)
    # txt = re.sub('\u00EC|\u00ED|\u00EE|\u00EF', 'i', txt)
    # txt = re.sub('\u00F0', 't', txt)
    # txt = re.sub('\u00F1', 'n', txt)
    # txt = re.sub('\u00F2|\u00F3|\u00F4|\u00F5|\u00F6', 'o', txt)
    # txt = re.sub('\u00F7', '/', txt)
    # txt = re.sub('\u00F8', '0', txt)
    # txt = re.sub('\u00F9|\u00FA|\u00FB|\u00FC', 'u', txt)
    # txt = re.sub('\u00FD', 'y', txt)
    # txt = re.sub('\u00FE', 's', txt)
    # txt = re.sub('\u00FF', 'y', txt)
    
    # Improvement VAD 1: there are spaces in the new annotated dataset
    # txt = re.sub('\s+$', '', txt)
    # if txt.find('  ') != -1:
    #     print('Error: double space')
    return txt

def process_train_location(txt):
    if txt == '':
        return ''
    else:
        return txt

def process_train_location2text(text, txtloc):
    txtloc_s, txtloc_e = txtloc.split(':')
    itxtloc_s = int(txtloc_s)
    itxtloc_e = int(txtloc_e)
    return text[itxtloc_s:itxtloc_e]

def stance2cate(text):
    text2cate = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    return text2cate[text]

def load_and_prepare_train(root=""):
    df = pd.read_csv(root + "AnnotatedTwitterID_wAspect_wOpinion_cAspect_wText_train.csv")

    df['text'] = df['text'].apply(process_text)
    df['stance'] = df['stance'].apply(clean_spaces4text)
    df['stance_cate'] = df['stance'].apply(stance2cate)
    df['clean_text'] = df['text'].apply(clean_spaces)

    # comment our for testing in main
    df['aspect_span'] = df['aspect_span'].apply(process_train_location)
    df['aspect_span_text'] = df.apply(lambda row : process_train_location2text(row['clean_text'],
                                      row['aspect_span']), axis=1)
    # print(df['aspect_span_text'].head)
    # df['target'] = ""
    return df

# ///////////////////// Load and prepare test
def load_and_prepare_test(root=""):
    df = pd.read_csv(root + "AnnotatedTwitterID_wAspect_wOpinion_cAspect_wText_test.csv")

    df['text'] = df['text'].apply(process_text)
    df['stance'] = df['stance'].apply(clean_spaces4text)
    df['stance_cate'] = df['stance'].apply(stance2cate)
    df['clean_text'] = df['text'].apply(clean_spaces)

    # comment our for testing in main
    df['aspect_span'] = df['aspect_span'].apply(process_train_location)
    df['aspect_span_text'] = df.apply(lambda row : process_train_location2text(row['clean_text'],
                                      row['aspect_span']), axis=1)
    return df


def process_test_data(clean_text, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=100, model_name=None):
    
    text = clean_text

    if 'albert-large-v2' in model_name:
        encoding = tokenizer(
            text,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            add_special_tokens=False)
        input_ids_text = encoding["input_ids"]
        text_offsets = encoding["offset_mapping"]
    else:
        assert False

    if input_ids_text[0] == special_tokens["cls"]:
        # getting rid of special tokens
        # if cls is in the sentence, means sep is in the entence as well
        input_ids_text = input_ids_text[1:-1]
        text_offsets = text_offsets[1:-1]

    sec1_text_ids = []
    
    new_max_len = max_len - 3 - len(sec1_text_ids)
    if new_max_len < len(input_ids_text):
        print('Warning: new_max_len %d < input_ids_text len %d' % (new_max_len, len(input_ids_text)))
    
    input_ids = (
        [special_tokens["cls"]] + sec1_text_ids + [special_tokens["sep"]]
            + input_ids_text[:new_max_len]
            + [special_tokens["sep"]]
    )

    token_type_ids = [0] + [0] * len(sec1_text_ids) + [0] + [1] * (len(input_ids_text[:new_max_len]) + 1)
    text_offsets = [(0, 0)] * (2 + len(sec1_text_ids)) + text_offsets[:new_max_len] + [(0, 0)]

    assert len(input_ids) == len(token_type_ids) 
    assert len(input_ids) == len(text_offsets), (len(input_ids), len(text_offsets))

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([special_tokens["pad"]] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)
    
    if np.max(text_offsets) > len(text):
        print('Warning: np.max(text_offsets) %d > len(text) len %d' % (np.max(text_offsets), len(text)))

    return {
        "ids": input_ids,
        "token_type_ids": token_type_ids,
        "text": text,
        "offsets": text_offsets,
    }


class TweetTestDataset(Dataset):
    def __init__(self, df, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=200, model_name="bert"):
        self.special_tokens = special_tokens
        self.precomputed_tokens_and_offsets = precomputed_tokens_and_offsets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

        self.text_external_ids = df['ID'].values
        self.clean_texts = df['clean_text'].values

    def __len__(self):
        return len(self.clean_texts)

    def __getitem__(self, idx):

        data = process_test_data(self.clean_texts[idx], self.tokenizer, self.special_tokens, self.precomputed_tokens_and_offsets,
                                 max_len=self.max_len, model_name=self.model_name)
        return {
            'external_id': self.text_external_ids[idx],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'text': data["text"],
            'offsets': np.array(data["offsets"], dtype=np.int_)
        }


def process_training_data(clean_text, aspect_span, aspect_span_text, stance, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=100, model_name=None, stance_as_feature=None):
    text = clean_text

    # chars is 0 except for those chars in text span
    chars = np.zeros((len(text)))
    idx = text.find(aspect_span_text)
    chars[idx:idx + len(aspect_span_text)] = 1
    
    s_aspect, e_aspect = aspect_span.split(':')
    i_s_aspect = int(s_aspect)
    i_e_aspect = int(e_aspect)
    assert i_s_aspect == idx
    assert i_e_aspect == idx + len(aspect_span_text)

    # if tokenizer.name:
    if 'albert' in model_name:
        input_ids_text = precomputed_tokens_and_offsets['ids'][text]
        text_offsets = precomputed_tokens_and_offsets['offsets'][text]
    else:
        assert False

    # Still need to get rid of special tokens. They will be manually added later.
    #  So both token ids and offsets are removed, and then added later
    if input_ids_text[0] == special_tokens["cls"]:
        # getting rid of special tokens
        # if cls is in the sentence, means sep is in the entence as well,
        #  This sep is different from the "adding an extra SEP after tokenization"
        input_ids_text = input_ids_text[1:-1]
        text_offsets = text_offsets[1:-1]

    # Pre-computed text needs trim of head and tail token
    if stance_as_feature is not None:
        # print('Is not none')
        sec1_text_ids = precomputed_tokens_and_offsets["ids"][stance]
        sec1_text_offsets = precomputed_tokens_and_offsets["offsets"][stance]
        if sec1_text_ids[0] == special_tokens["cls"]:
            sec1_text_ids = sec1_text_ids[1:-1]
            sec1_text_offsets = sec1_text_offsets[1:-1]
    else:
        sec1_text_ids = []
        sec1_text_offsets = []

    # So the new is CLS + SEP + text_ids + SEP
    if 'deberta' in model_name:
        assert len(sec1_text_ids) <= 13
    else:
        assert len(sec1_text_ids) <= 16
    new_max_len = max_len - 3 - len(sec1_text_ids)
    if new_max_len < len(input_ids_text):
        print('new_max_len {}, max_len {}, len(sec1_text_ids) {}, len(input_ids_text) {}'.format(
            new_max_len, max_len, len(sec1_text_ids), len(input_ids_text)))
    if new_max_len < len(input_ids_text):
        print('Warning: new_max_len %d < input_ids_text len %d' % (new_max_len, len(input_ids_text)))
    input_ids = (
        [special_tokens["cls"]] + sec1_text_ids + [special_tokens["sep"]]
            + input_ids_text[:new_max_len]
            + [special_tokens["sep"]]
    )
    # token_type_ids indicates whether the token is special or non-special
    # Be aware that the last sep is 1 according to https://huggingface.co/transformers/v3.2.0/glossary.html
    # token_type_ids = [0, 0, 0] + [1] * (len(input_ids_text[:new_max_len]) + 1)
    token_type_ids = [0] + [0] * len(sec1_text_ids) + [0] + [1] * (len(input_ids_text[:new_max_len]) + 1)
    # The special tokens correspond to no offsets, as well as the last sep

    text_offsets = [(0, 0)] * (2 + len(sec1_text_ids)) + text_offsets[:new_max_len] + [(0, 0)]
    assert len(input_ids) == len(token_type_ids) 
    assert len(input_ids) == len(text_offsets), (len(input_ids), len(text_offsets))

    # Padding is meant for those too short. But too long sentences will suffer from loss of data
    padding_length = max_len - len(input_ids)
    len_input_ids_before_padding_length = len(input_ids)
    # # To accommodate softmax
    # assert input_ids[len_input_ids_before_padding_length - 1] ==  special_tokens["sep"]
    # # To accommodate softmax
    if padding_length > 0:
        input_ids = input_ids + ([special_tokens["pad"]] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        text_offsets = text_offsets + ([(0, 0)] * padding_length)

    toks = []

    for i, (a, b) in enumerate(text_offsets):
        # each offset, calculate the sum. If sum > 0, which means that the span at least has one character in the ground-truth span (it can be a char following several spaces), then the span will be annotated as 'included'
        sm = np.sum(chars[a:b])
        if sm > 0:
            toks.append(i)
    if len(toks) == 0:
        toks = [0]
        
    targets_start = toks[0]
    targets_end = toks[-1]
    
    aspect_span_input_ids = (
        [special_tokens["cls"], special_tokens["sep"]]
            + input_ids_text[targets_start: targets_end + 1]
            + [special_tokens["sep"]]
    )
    aspect_span_token_type_ids = [0, 0] + [1] * (len(input_ids_text[targets_start: targets_end + 1]) + 1)

    padding_length_aspect_span = max_len - len(aspect_span_input_ids)
    if padding_length_aspect_span > 0:
        aspect_span_input_ids = aspect_span_input_ids + ([special_tokens["pad"]] * padding_length_aspect_span)
        aspect_span_token_type_ids = aspect_span_token_type_ids + ([0] * padding_length_aspect_span)
    
    return {
        "ids": input_ids,
        "token_type_ids": token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        "text": text,
        'aspect_span': aspect_span,
        'aspect_span_text': aspect_span_text,
        'aspect_span_ids': aspect_span_input_ids,
        'aspect_span_token_type_ids': aspect_span_token_type_ids,
        "offsets": text_offsets,
    }


np_str_obj_array_pattern = re.compile(r'[SaUO]')

import collections
from torch._six import string_classes

vad_collate_err_msg_format = (
    "vad_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class vad_collate:
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    def __init__(self, tokenizer, mlm_probability):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(vad_collate_err_msg_format.format(elem.dtype))
                return self.__call__([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            try:
                batch_dict = {key: self.__call__([d[key] for d in batch]) for key in elem}
                labels = batch_dict['ids'].clone()
                inputs = batch_dict['ids'].clone()
                # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
                probability_matrix = torch.full(labels.shape, self.mlm_probability)
                if 'special_tokens_mask' not in batch_dict:
                    special_tokens_mask = [
                        self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                    ]
                    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
                else:
                    special_tokens_mask = special_tokens_mask.bool()

                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = -100  # We only compute loss on masked tokens

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
                inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

                # 10% of the time, we replace masked input tokens with random word
                indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                inputs[indices_random] = random_words[indices_random]
                batch_dict['inputs'] = inputs
                batch_dict['labels'] = labels
                return elem_type(batch_dict)
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: self.__call__([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.__call__(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [self.__call__(samples) for samples in transposed]  # Backwards compatibility.
            else:
                try:
                    return elem_type([self.__call__(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self.__call__(samples) for samples in transposed]

        raise TypeError(vad_collate_err_msg_format.format(elem_type))


# TweetTrainDataset always focuses on the token level, and only uses CrossEntropy loss
# Metric is used only in the validation set.
class TweetTrainingDataset(Dataset):
    def __init__(self, df, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=310, model_name="bert"):
        self.special_tokens = special_tokens
        self.precomputed_tokens_and_offsets = precomputed_tokens_and_offsets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

        self.text_external_ids = df['ID'].values
        self.clean_texts = df['clean_text'].values

        self.aspect_spans = df['aspect_span'].values
        self.aspect_span_texts = df['aspect_span_text'].values
        
        self.stances = df['stance'].values
        self.stances_cate = df['stance_cate'].values

    def __len__(self):
        return len(self.clean_texts)

    def __getitem__(self, idx):

        data = process_training_data(self.clean_texts[idx], self.aspect_spans[idx], self.aspect_span_texts[idx], self.stances[idx], self.tokenizer, self.special_tokens, self.precomputed_tokens_and_offsets,
                                     max_len=self.max_len, model_name=self.model_name)

        return {
            'external_id': self.text_external_ids[idx],
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': data["targets_start"],
            'targets_end': data["targets_end"],
            'text': data["text"], 
            "aspect_span": data['aspect_span'],
            "aspect_span_texts": data["aspect_span_text"],
            "aspect_span_ids": torch.tensor(data["aspect_span_ids"], dtype=torch.long),
            "aspect_span_token_type_ids": torch.tensor(data["aspect_span_token_type_ids"], dtype=torch.long),
            'stance_cate': self.stances_cate[idx],
            'offsets': np.array(data["offsets"], dtype=np.int_)
        }


class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.best_start = None
        self.best_end = None

    def __call__(self, epoch_score, model, model_path, start_oof, end_oof):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.best_start = start_oof
            self.best_end = end_oof
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_start = start_oof
            self.best_end = end_oof
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), model_path)
        self.val_score = epoch_score


def loss_fn(start_logits, end_logits, start_positions, end_positions,clss, stance_cate, kld_loss, lm_loss, weight=None):
    if weight is not None:
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        start_loss = loss_fct(start_logits, start_positions)
        start_loss = (start_loss * weight / weight.sum()).sum()
        start_loss = start_loss.mean()
        end_loss = loss_fct(end_logits, end_positions)
        end_loss = (end_loss * weight / weight.sum()).sum()
        end_loss = end_loss.mean()
        cls_loss = loss_fct(clss, stance_cate).mean()
    else:
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        cls_loss = loss_fct(clss, stance_cate)

    total_loss = start_loss + end_loss + cls_loss + kld_loss
    return total_loss


def token_pred_to_char_pred(token_pred, offsets):
    '''
    from token_predicted prob to char_pred prob
    '''

    if len(token_pred.shape) == 1:
        char_pred = np.zeros(np.max(offsets))
    else:
        char_pred = np.zeros((np.max(offsets), token_pred.shape[1]))
    for i in range(len(token_pred)):
        s, e = int(offsets[i][0]), int(offsets[i][1])  # start, end of an offset
        char_pred[s:e] = token_pred[i] # The char becomes a value between 0~1

    return char_pred


def char_target_to_span(char_target):
    spans = []
    start, end = 0, 0
    for i in range(len(char_target)):
        if char_target[i] == 1 and char_target[i - 1] == 0:
            if end:
                spans.append([start, end])
            start = i
            end = i + 1
        elif char_target[i] == 1:
            end = i + 1
        else:
            if end:
                spans.append([start, end])
            start, end = 0, 0
    if end == len(char_target) and start != 0:
        # The last one is included
        spans.append([start, end])
        start = 0
        end = 0
    assert start == 0, end == 0
    return spans


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MicroF1Meter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_TP = 0
        self.sum_FN = 0
        self.sum_FP = 0
        self.prec = 0
        self.reca = 0
        self.microF1 = 0
        self.count = 0

    def update(self, val_TP, val_FN, val_FP, n=1):
        # self.val = val_TP
        self.sum_TP += val_TP
        self.sum_FN += val_FN
        self.sum_FP += val_FP
        self.count += n
        # self.avg = self.sum / self.count
        if self.sum_TP == 0:
            self.prec = 0
            self.reca = 0
            self.microF1 = 0
        else:
            self.prec = self.sum_TP / (self.sum_TP + self.sum_FP)
            self.reca = self.sum_TP / (self.sum_TP + self.sum_FN)
            self.microF1 = 2 * self.prec * self.reca / (self.prec + self.reca)


# if the two intervals overlaps
def is_overlaping(a, b):
  if b[0] >= a[0] and b[0] < a[1]:
    return True
  else:
    return False


def merge_two_sorted_spans(spans_pred, spans_grt):
    '''
    After the merge, the resulting interval list might have overlapping spans. Thus we need to call part of 
    'merge_locations' code to merge them.
    '''
    test_list1 = spans_pred
    test_list2 = spans_grt
    size_1 = len(test_list1)
    size_2 = len(test_list2) 
    res = []
    i, j = 0, 0

    while i < size_1 and j < size_2:
        if test_list1[i][0] < test_list2[j][0]: 
            res.append(test_list1[i]) 
            i += 1
        else:
            res.append(test_list2[j]) 
            j += 1
    res = res + test_list1[i:] + test_list2[j:]

    arr = res
    # sort the intervals by its first value
    arr.sort(key = lambda x: x[0])

    merged_list= []
    if len(arr) > 0:
        merged_list.append(arr[0])
        for i in range(1, len(arr)):
            pop_element = merged_list.pop()
            if is_overlaping(pop_element, arr[i]):
                new_element = [pop_element[0], max(pop_element[1], arr[i][1])]
                merged_list.append(new_element)
            else:
                merged_list.append(pop_element)
                merged_list.append(arr[i])
    return merged_list


def intersect_two_sorted_spans(spans_pred, spans_grt):
    # i and j pointers for arr1
    # and arr2 respectively
    i = j = 0
    
    n = len(spans_pred)
    m = len(spans_grt)

    # Loop through all intervals unless one
    # of the interval gets exhausted
    res = []
    while i < n and j < m:
        
        # Left bound for intersecting segment
        l = max(spans_pred[i][0], spans_grt[j][0])
        
        # Right bound for intersecting segment
        r = min(spans_pred[i][1], spans_grt[j][1])
        
        # If segment is valid print it
        # if l <= r:
        if l < r:
            # since the (0, 5) and (5, 6) should have no overlap
            # print('{', l, ',', r, '}')
            res.append([l, r])

        # If i-th interval's right bound is
        # smaller increment i else increment j
        if spans_pred[i][1] < spans_grt[j][1]:
            i += 1
        else:
            j += 1
    return res


def calculate_TP_FN_FP_between_2lstsofintervals(
    text,
    offsets,
    spans_pred,
    spans_grt):
    char_pred = np.zeros(len(text))
    char_grt = np.zeros(len(text))
    tp = 0
    fn = 0
    fp = 0
    for aspan_pred in spans_pred:
        a = offsets[aspan_pred[0]][0]
        # if aspan_pred[1] > 120:
        #     print(b)
        b = offsets[aspan_pred[1]][1]
        char_pred[offsets[aspan_pred[0]][0]:offsets[aspan_pred[1]][1]] = 1
    for aspan_grt in spans_grt:
        char_grt[aspan_grt[0]:aspan_grt[1]] = 1
        # if is_overlaping(aspan_pred, aspan_grt):
        #     print('Overlapping detected!')
    for i in range(len(text)):
        if char_pred[i] == 1 and char_grt[i] == 1:
            tp += 1
        elif char_grt[i] == 1:
            fn += 1
        elif char_pred[i] == 1:
            fp += 1
    return tp, fn, fp


def calculate_preds_between_2lstsofintervals(
    text,
    offsets,
    spans_pred,
    spans_grt):
    char_pred = np.zeros(len(text))
    char_grt = np.zeros(len(text))
    # spans to binary
    for aspan_pred in spans_pred:
        char_pred[offsets[aspan_pred[0]][0]:offsets[aspan_pred[1]][1]] = 1
    for aspan_grt in spans_grt:
        char_grt[aspan_grt[0]:aspan_grt[1]] = 1

    char_pred_trimmed = char_pred # np.zeros(len(text))
    char_grt_trimmed = char_grt # np.zeros(len(text))

    return char_pred_trimmed, char_grt_trimmed


def eval_fn(data_loader, model, device, model_name, activation):
    model.eval()
    losses = AverageMeter()
    mf1 = MicroF1Meter()
    mf1_se = MicroF1Meter()
    mf1_st = MicroF1Meter()

    bin_preds_byb = []
    bin_truths_byb = []

    start_array, end_array = [], []

    ext_ids_byb = []
    locations_byb = []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            # mask = mask.to(device, dtype=torch.long)

            clean_text = d["text"]
            aspect_span_ids = d["aspect_span_ids"]
            aspect_span_token_type_ids = d["aspect_span_token_type_ids"]
            aspect_span_ids = aspect_span_ids.to(device, dtype=torch.long)
            aspect_span_token_type_ids = aspect_span_token_type_ids.to(device, dtype=torch.long)
            lm_inputs = d["inputs"].to(device, dtype=torch.long)
            lm_labels = d["labels"].to(device, dtype=torch.long)
            B, SEQ = ids.size()
            ids, token_type_ids, lm_inputs, lm_labels = trim_tensors(
                ids, token_type_ids, lm_inputs, lm_labels, model_name
            )
            aspect_span_ids, aspect_span_token_type_ids, _, _ = trim_tensors(
                aspect_span_ids, aspect_span_token_type_ids, lm_inputs, lm_labels, model_name
            )

            outputs_start, outputs_end, clss, kld_loss, lm_loss = model(
                # ids=ids,
                tokens=ids,
                token_type_ids=token_type_ids,
                aspect_tokens=aspect_span_ids,
                aspect_token_type_ids=aspect_span_token_type_ids,
                lm_inputs=lm_inputs,
                lm_labels=lm_labels)
            
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            stance_cate = d["stance_cate"]

            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            stance_cate = stance_cate.to(device, dtype=torch.long)

            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, clss, stance_cate, kld_loss, lm_loss, None)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            clss = torch.softmax(clss, dim=1).cpu().detach().numpy()

            outputs_start4start_array_concat = np.zeros((B, SEQ))
            outputs_end4start_array_concat = np.zeros((B, SEQ))
            _, SEQ_TRIM = outputs_start.shape
            outputs_start4start_array_concat[:, :SEQ_TRIM] = outputs_start
            outputs_end4start_array_concat[:, :SEQ_TRIM] = outputs_end
            start_array.append(outputs_start4start_array_concat)
            end_array.append(outputs_end4start_array_concat)

            location = d["aspect_span"]

            bin_preds = []
            bin_truths = []

            ext_ids = d["external_id"]
            ext_ids_byb.extend(ext_ids)
            locations_byb.extend(location)
            tps = []
            fns = []
            fps = []
            tps_se = 0
            fns_se = 0
            fps_se = 0
            tps_st = 0
            fns_st = 0
            fps_st = 0
            for px, clean_text_a in enumerate(clean_text):
                start_end_pair = []
                for start_index in np.argsort(outputs_start[px, :])[::-1][:20]:
                    # Here, also the top20 end probabilities
                    for end_index in np.argsort(outputs_end[px, :])[::-1][:20]:
                        if end_index < start_index:
                            continue
                        start_end_pair.append((start_index, end_index))
                if len(start_end_pair) == 0:
                    # start_end_pair.append((-1, 0))
                    # start_end_pair = []
                    spans_pred = []
                else:
                    idx_start = start_end_pair[0][0]
                    idx_end = start_end_pair[0][1]
                    spans_pred = [(idx_start, idx_end)]
                s_s, s_e = location[px].split(':')
                i_s = int(s_s)
                i_e = int(s_e)
                spans_grt = [(i_s, i_e)]

                tp, fn, fp = calculate_TP_FN_FP_between_2lstsofintervals(
                    clean_text_a,
                    offsets=offsets[px],
                    spans_pred=spans_pred,
                    spans_grt=spans_grt
                )
                tps.append(tp)
                fns.append(fn)
                fps.append(fp)

                if offsets[px][spans_pred[0][0]][0] == spans_grt[0][0]:
                    tps_se += 1
                else:
                    fps_se += 1
                    fns_se += 1
                if offsets[px][spans_pred[0][1]][1] == spans_grt[0][1]:
                    tps_se += 1
                else:
                    fps_se += 1
                    fns_se += 1
                stance_pred = np.argmax(clss[px, :])
                if stance_pred == stance_cate[px].item():
                    tps_st += 1
                else:
                    fns_st += 1
                    fps_st += 1

                pred, truth = calculate_preds_between_2lstsofintervals(
                    clean_text_a,
                    offsets=offsets[px],
                    spans_pred=spans_pred,
                    spans_grt=spans_grt
                )

                if not len(pred) and not len(truth):
                    print('Warning: not len(pred) and not len(truth) not satisified')
                    preds_for_f1score_input = []
                    truth_for_f1score_input = []
                else:
                    assert len(pred) == len(truth)
                    bin_preds.append(pred)
                    bin_truths.append(truth)
                    preds_for_f1score_input = np.concatenate(bin_preds)
                    truth_for_f1score_input = np.concatenate(bin_truths)

            preds_for_f1score_input = np.concatenate(bin_preds)
            truth_for_f1score_input = np.concatenate(bin_truths)
            bin_preds_byb.extend(preds_for_f1score_input)
            bin_truths_byb.extend(truth_for_f1score_input)
            
            if bi % 50 == 0:
                micro_F1_sklearn_byb = f1_score(bin_preds_byb, bin_truths_byb)
            mf1.update(np.sum(tps), np.sum(fns), np.sum(fps), ids.size(0))
            mf1_se.update(tps_se, fns_se, fps_se, ids.size(0))
            mf1_st.update(tps_st, fns_st, fps_st, ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, mF1=mf1.microF1, mF1_se=mf1_se.microF1, mf1_st=mf1_st.microF1)
    start_array = np.concatenate(start_array)
    end_array = np.concatenate(end_array)
    print(f"micro_F1 = {mf1.microF1}")
    micro_F1_sklearn_byb = f1_score(bin_preds_byb, bin_truths_byb)
    print(f"mF1sklbyb = {micro_F1_sklearn_byb}")
    return mf1.microF1, start_array, end_array


def train_fn(data_loader, model, optimizer, device, scheduler=None, opt=None, epc=None, tokenizer=None, model_name=None, activation='sigmoid'):
    # model.name
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = AverageMeter()
    mf1 = MicroF1Meter()
    mf1_se = MicroF1Meter()
    mf1_st = MicroF1Meter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    print('for bi, d in enumerate(tk0):')
    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        clean_text = d["text"]

        stance_cate = d["stance_cate"]

        orig_selected = d["aspect_span_texts"]
        aspect_span = d["aspect_span"]
        aspect_span_ids = d["aspect_span_ids"]
        aspect_span_token_type_ids = d["aspect_span_token_type_ids"]
        offsets = d["offsets"]
        offsets = offsets.detach().numpy()

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        stance_cate = stance_cate.to(device, dtype=torch.long)

        aspect_span_ids = aspect_span_ids.to(device, dtype=torch.long)
        aspect_span_token_type_ids = aspect_span_token_type_ids.to(device, dtype=torch.long)
        
        lm_inputs = d["inputs"].to(device, dtype=torch.long)
        lm_labels = d["labels"].to(device, dtype=torch.long)

        model.zero_grad(set_to_none=True)

        ids, token_type_ids, lm_inputs, lm_labels = trim_tensors(
            ids, token_type_ids, lm_inputs, lm_labels, model_name
        )
        aspect_span_ids, aspect_span_token_type_ids, _, _ = trim_tensors(
            aspect_span_ids, aspect_span_token_type_ids, lm_inputs, lm_labels, model_name
        )
        outputs_start, outputs_end, clss, kld_loss, lm_loss = model(
            tokens=ids,
            token_type_ids=token_type_ids,
            aspect_tokens=aspect_span_ids,
            aspect_token_type_ids=aspect_span_token_type_ids,
            lm_inputs=lm_inputs,
            lm_labels=lm_labels
        )

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end, clss, stance_cate, kld_loss, lm_loss, None)
        
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        clss = torch.softmax(clss, dim=1).cpu().detach().numpy()

        tps = []
        fns = []
        fps = []
        tps_se = 0
        fns_se = 0
        fps_se = 0
        tps_st = 0
        fns_st = 0
        fps_st = 0

        for px, clean_text_a in enumerate(clean_text):

            start_end_pair = []

            for start_index in np.argsort(outputs_start[px, :])[::-1][:20]:
                # Here, also the top20 end probabilities
                for end_index in np.argsort(outputs_end[px, :])[::-1][:20]:
                    if end_index < start_index:
                        continue
                    start_end_pair.append((start_index, end_index))
            if len(start_end_pair) == 0:
                # start_end_pair.append((-1, 0))
                # start_end_pair = []
                spans_pred = []
            else:
                idx_start = start_end_pair[0][0]
                idx_end = start_end_pair[0][1]
                spans_pred = [(idx_start, idx_end)]
            s_s, s_e = aspect_span[px].split(':')
            i_s = int(s_s)
            i_e = int(s_e)
            spans_grt = [(i_s, i_e)]

            tp, fn, fp = calculate_TP_FN_FP_between_2lstsofintervals(
                clean_text_a,
                offsets=offsets[px],
                spans_pred=spans_pred,
                spans_grt=spans_grt
            )

            if offsets[px][spans_pred[0][0]][0] == spans_grt[0][0]:
                tps_se += 1
            else:
                fps_se += 1
                fns_se += 1
            if offsets[px][spans_pred[0][1]][1] == spans_grt[0][1]:
                tps_se += 1
            else:
                fps_se += 1
                fns_se += 1

            tps.append(tp)
            fns.append(fn)
            fps.append(fp)

            stance_pred = np.argmax(clss[px, :])
            if stance_pred == stance_cate[px].item():
                tps_st += 1
            else:
                fns_st += 1
                fps_st += 1

        mf1.update(np.sum(tps), np.sum(fns), np.sum(fps), ids.size(0))
        mf1_se.update(tps_se, fns_se, fps_se, ids.size(0))
        mf1_st.update(tps_st, fns_st, fps_st, ids.size(0))
        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg, micro_F1=mf1.microF1, mF1_se=mf1_se.microF1, mF1_st=mf1_st.microF1)


def run_a_trval(config, fold, dfx, tr, val, freeze_weight=None):
    df_train = dfx.iloc[tr]
    df_valid = dfx.iloc[val]
    set_seed(SEED)
    tokenizer, special_tokens, precomputed_tokens_and_offsets = create_tokenizer_and_tokens(dfx, config)
    
    print('TweetTrainingDataset')
    train_dataset = TweetTrainingDataset(df_train, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=config.MAX_LEN, model_name=config.selected_model)
    train_sampler = RandomSampler(train_dataset)
    vad_collator = vad_collate(tokenizer=tokenizer, mlm_probability=0.15)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.BATCH_SIZE_TR,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=vad_collator
    )

    valid_dataset = TweetTrainingDataset(df_valid, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=config.MAX_LEN, model_name=config.selected_model)

    eval_sampler = SequentialSampler(valid_dataset)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=eval_sampler,
        batch_size=config.batch_size_val,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=vad_collator
    )

    print('VADTransformer')
    model = VADTransformer(
        config.selected_model,
        nb_layers=config.nb_layers,
        nb_ft=config.nb_ft,
        nb_class=config.nb_class,
        pretrained=config.pretrained,
        nb_cate=config.nb_cate,
        multi_sample_dropout=config.multi_sample_dropout,
        training=True
    )

    device = torch.device("cuda")

    model.to(device)
    if freeze_weight:
        model.load_state_dict(torch.load(freeze_weight))
    print('model.to(device)')
    num_train_steps = int(len(df_train) / config.BATCH_SIZE_TR * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.LR)

    scheduler = None
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_steps // 20,
        # num_warmup_steps=num_train_steps // 100,
        num_training_steps=num_train_steps
    )

    model = torch.nn.DataParallel(model)

    # This self-defined class handle the early stopping state
    es = EarlyStopping(patience=3, mode="max", delta=0.0005)
    print(f"Training is Starting for fold={fold}")

    set_seed(SEED)

    for epoch in range(config.EPOCHS):
        # Each epoch's training.
        print('====== Start entering train_fn of epoch %d' % (epoch))
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler, epc=epoch, tokenizer=tokenizer, model_name=config.selected_model, activation=config.activation)
        print('train_fn')
        mf1, startoof, endoof = eval_fn(valid_data_loader, model, device, model_name=config.selected_model, activation=config.activation)
        es(mf1, model, model_path=f"{config.out_path}/ckpt_{config.selected_model}_{fold}.bin", start_oof=startoof, end_oof=endoof)
        if es.early_stop:
            print("Early stopping")
            break
    del model, optimizer

    gc.collect()

    return es.best_score, len(valid_dataset), es.best_start, es.best_end


def predict(model, dataset, batch_size=32, num_workers=1, activation='sigmoid'):
    model.eval()
    pred_clss = []
    start_probas = []
    end_probas = []

    # batch_size=32
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for data in loader:
            ids, token_type_ids, _, _ = trim_tensors(
                data["ids"], data["token_type_ids"], data["ids"], data["token_type_ids"], model.name
            )
            aspect_span_ids, aspect_span_token_type_ids, _, _ = trim_tensors(
                data["aspect_span_ids"], data["aspect_span_token_type_ids"], data["ids"], data["token_type_ids"], model.name
            )
            
            outputs_start, outputs_end, clss, _, _ = model(
                ids.cuda(), token_type_ids.cuda(),
                aspect_tokens=aspect_span_ids.cuda(), aspect_token_type_ids=aspect_span_token_type_ids.cuda()
            )
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            clss = torch.softmax(clss, dim=1).cpu().detach().numpy()
            
            start_probas.extend(list(outputs_start))
            end_probas.extend(list(outputs_end))
            pred_clss.extend(list(clss))

    return start_probas, end_probas, pred_clss


def k_fold_inference(config, test_dataset, seed=42):

    seed_everything(seed)
    pred_tests = []
    print('config.nb_layers in {} is {}'.format(config.selected_model, config.nb_layers))
    print('and the config.name in %s is %s' % (config.selected_model, config.name))
    for weight in config.weights:
        model = VADTransformer(
            config.selected_model,
            nb_layers=config.nb_layers,
            nb_ft=config.nb_ft,
            nb_class=config.nb_class,
            nb_cate=config.nb_cate,
            multi_sample_dropout=config.multi_sample_dropout,
        ).cuda()
        model = load_model_weights(model, weight, cp_folder=config.weights_path, verbose=1)

        model.zero_grad(set_to_none=True)

        pred_test_afold = predict(model, test_dataset, batch_size=config.batch_size_val, num_workers=2, activation=config.activation)
        pred_tests.append(pred_test_afold)
    print('nb of weights', len(pred_tests))
    print('nb of instances', len(pred_tests[0][0]))
    print('nb of instances', len(pred_tests[0][1]))
    # a softmax over categories
    print('size of a cls', len(pred_tests[0][2][0]))
    # a softmax over locations
    print('size of a start', len(pred_tests[0][0][0]))
    print('size of an end', len(pred_tests[0][1][0]))
    return pred_tests


def get_char_preds_eval(pred_tests, train_dataset, fpathout):
    mf1 = MicroF1Meter()
    mf1_se = MicroF1Meter()
    mf1_st = MicroF1Meter()
    pred_test_clss = []
    char_pred_test_start = []
    char_pred_test_end = []
    lst_for_csv_output = []
    tps = []
    fns = []
    fps = []
    tps_se = 0
    fns_se = 0
    fps_se = 0
    tps_st = 0
    fns_st = 0
    fps_st = 0
    for idx in range(len(train_dataset)):
        # instead of next
        d = train_dataset[idx]
        # print(d)
        text = d['text']
        # print(len(text))
        offsets = d['offsets']
        external_id = d['external_id']
        stance_cate = d["stance_cate"]
        location = d["aspect_span"]

        start_preds = np.mean([pred_tests[i][0][idx] for i in range(len(pred_tests))], 0)
        end_preds = np.mean([pred_tests[i][1][idx] for i in range(len(pred_tests))], 0)
        cls_preds = np.mean([pred_tests[i][2][idx] for i in range(len(pred_tests))], 0)
        start_pred = np.argmax(start_preds)
        end_pred = np.argmax(end_preds)
        pred_test_clss.append(cls_preds)
        st_pred = np.argmax(cls_preds)

        lst_for_csv_output.append(
            [external_id, text, st_pred, text[offsets[start_pred][0]:offsets[end_pred][1]]])
        
        spans_pred = [(start_pred, end_pred)]
        s_s, s_e = location.split(':')
        i_s = int(s_s)
        i_e = int(s_e)
        spans_grt = [(i_s, i_e)]
        tp, fn, fp = calculate_TP_FN_FP_between_2lstsofintervals(
            text,
            offsets=offsets,
            spans_pred=spans_pred,
            spans_grt=spans_grt
        )
        tps.append(tp)
        fns.append(fn)
        fps.append(fp)

        if offsets[spans_pred[0][0]][0] == spans_grt[0][0]:
            tps_se += 1
        else:
            fps_se += 1
            fns_se += 1
        if offsets[spans_pred[0][1]][1] == spans_grt[0][1]:
            tps_se += 1
        else:
            fps_se += 1
            fns_se += 1
        if st_pred == stance_cate:
            tps_st += 1
        else:
            fns_st += 1
            fps_st += 1
    mf1.update(np.sum(tps), np.sum(fns), np.sum(fps), len(train_dataset))
    mf1_se.update(tps_se, fns_se, fps_se, len(train_dataset))
    mf1_st.update(tps_st, fns_st, fps_st, len(train_dataset))
    print('mf1_st=%f, mf1_se=%f, mf1_span=%f' % (mf1_st.microF1, mf1_se.microF1, mf1.microF1))
    fp = open(fpathout, 'wt', encoding='utf8')
    csv_writer = csv.writer(fp, lineterminator='\n')
    csv_writer.writerow(["ID", "text", "stance", "aspect_span"])
    csv_writer.writerows(lst_for_csv_output)
    fp.close()
    return pred_test_clss, char_pred_test_start, char_pred_test_end


def get_char_preds_output(pred_tests, test_dataset, fpathout):
    pred_test_clss = []
    char_pred_test_start = []
    char_pred_test_end = []
    lst_for_csv_output = []
    for idx in range(len(test_dataset)):
        # instead of next
        d = test_dataset[idx]
        text = d['text']
        offsets = d['offsets']
        external_id = d['external_id']

        start_preds = np.mean([pred_tests[i][0][idx] for i in range(len(pred_tests))], 0)
        end_preds = np.mean([pred_tests[i][1][idx] for i in range(len(pred_tests))], 0)
        cls_preds = np.mean([pred_tests[i][2][idx] for i in range(len(pred_tests))], 0)

        start_pred = np.argmax(start_preds)
        end_pred = np.argmax(end_preds)
        pred_test_clss.append(cls_preds)
        st_pred = np.argmax(cls_preds)
        lst_for_csv_output.append(
            [external_id, text, st_pred, text[offsets[start_pred][0]:offsets[end_pred][1]]])

    fp = open(fpathout, 'wt', encoding='utf8')
    csv_writer = csv.writer(fp, lineterminator='\n')
    csv_writer.writerow(["ID", "text", "stance", "aspect_span"])
    csv_writer.writerows(lst_for_csv_output)
    fp.close()
    return pred_test_clss, char_pred_test_start, char_pred_test_end


class ConfigVAD:
    selected_model = "vadlm-albert-large-v2"
    pretrained = True
    lowercase = True
    nb_layers = 8
    nb_ft = 128
    nb_class = 2
    nb_cate = 3
    multi_sample_dropout = True
    use_old_sentiment = False

    # training
    activation = "sigmoid"

    MAX_LEN = 120
    # BATCH_SIZE_TR = 16 # 16 best
    # BATCH_SIZE_TR = 24
    # BATCH_SIZE_TR = 4
    BATCH_SIZE_TR = 8
    EPOCHS = 10
    # LR = 2e-5
    LR = 3e-5
    # LR = 2e-11

    # Inference, it's also used in validation
    batch_size_val = 8
    max_len_val = 120
    num_workers = 8
    pin_memory = True

    out_path = f"../datasets/vadcheckpoints/5-fold-211103/{selected_model}"
    weights_path = f'../datasets/vadcheckpoints/5-fold-211103/{selected_model}'
    weights = None
    name = 'vadlm-albert-large-v2-squad'

    def __init__(self):
        if os.path.exists(self.weights_path):

            self.weights = sorted([f for f in os.listdir(self.weights_path) if "vadlm-albert-large-v2" in f])

def main():
    # ============ Begins the training and testing script
    torch.multiprocessing.set_start_method('spawn')
    seed_everything(SEED)
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # # ========== This is for training

    df_train = load_and_prepare_train(root=DATA_PATH)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)

    val_score = 0
    configs = [
        ConfigVAD(),
    ]
    config = configs[0]

    start_oof = np.zeros((len(df_train), config.MAX_LEN))
    end_oof = np.zeros((len(df_train), config.MAX_LEN))
    for i, (tr, val) in enumerate(skf.split(df_train, df_train.stance)):
        print('!!!!!!!!!!!!!!!!!!!! Training in fold %d' % (i))
        score_fold, cnt, start_logits_fold, end_logits_fold = run_a_trval(config, i, df_train, tr, val)
        val_score += score_fold
        start_oof[val, :] = start_logits_fold
        end_oof[val, :] = end_logits_fold

    # # ====================== For inference
    # preds = {}
    # df_test = load_and_prepare_test(root=DATA_PATH)
    # configs = [
    #     ConfigVAD(),
    # ]
    # for config in configs:
    #     # This is the config for vad, since there is only one config
    #     print(f'\n   -  Doing inference for {config.name}\n')

    #     tokenizer, special_tokens, precomputed_tokens_and_offsets = create_tokenizer_and_tokens(df_test, config)

    #     train_dataset = TweetTrainingDataset(df_test, tokenizer, special_tokens, precomputed_tokens_and_offsets, max_len=config.max_len_val, model_name=config.selected_model)
    #     pred_tests = k_fold_inference(
    #         config,
    #         train_dataset,
    #         seed=SEED,
    #     )

    #     print('Begin outputting PREDs.csv')
    #     char_start_end_cls = get_char_preds_eval(
    #         pred_tests,
    #         train_dataset,
    #         fpathout='st_and_aspectspan_preds.csv')
    #     preds[config.selected_model] = char_start_end_cls
    #     print('Finished outputting PREDS.csv')

    #     with open(f'{config.selected_model}-char_pred_test_logits_sgmd.pkl', 'wb') as handle:
    #         pickle.dump(char_start_end_cls, handle)


if __name__ == '__main__':
    main()

