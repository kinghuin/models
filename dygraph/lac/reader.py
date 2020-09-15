#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The file_reader converts raw corpus to input.
"""

import os
import argparse
import __future__
import io
import glob
import paddle
import numpy as np

def load_kv_dict(dict_path,
                 reverse=False,
                 delimiter="\t",
                 key_func=None,
                 value_func=None):
    """
    Load key-value dict from file
    """
    result_dict = {}
    for line in io.open(dict_path, "r", encoding='utf8'):
        terms = line.strip("\n").split(delimiter)
        if len(terms) != 2:
            continue
        if reverse:
            value, key = terms
        else:
            key, value = terms
        if key in result_dict:
            raise KeyError("key duplicated with [%s]" % (key))
        if key_func:
            key = key_func(key)
        if value_func:
            value = value_func(value)
        result_dict[key] = value
    return result_dict

class Dataset(paddle.io.Dataset):
    def __init__(self, args, mode, max_seq_len, _word_to_ids, _label_to_ids):
        self.mode = mode
        self.max_seq_len = max_seq_len

        self.word_ids = []
        self.label_ids = []
        with io.open(eval("args.%s_data"%mode), "r", encoding="utf-8") as fread:
            self.total = 0
            if mode == "infer":
                for line in fread:
                    words= line.strip()
                    self.word_ids.append(_word_to_ids(words))
                    self.total += 1
            else:
                headline = next(fread)
                for line in fread:
                    words, labels = line.strip("\n").split("\t")
                    if len(words)<1:
                        continue
                    self.word_ids.append(_word_to_ids(words.split("\002")))
                    self.label_ids.append(_label_to_ids(labels.split("\002")))
                    self.total += 1

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if self.mode == "infer":
            return self.word_ids[idx]
        else:
            return [self.word_ids[idx], self.label_ids[idx]]

    def padding_batch(self, batch):
        max_seq_len = min(max([len(sample[0]) for sample in batch]), self.max_seq_len)
        batch_word_ids = []
        batch_label_ids = []
        batch_lens = []
        

        for i, sample in enumerate(batch):
            sample_word_ids =  sample[0][:max_seq_len]
            sample_words_len = len(sample_word_ids)
            sample_word_ids += [0 for _ in range(max_seq_len-sample_words_len)]
            batch_word_ids.append(sample_word_ids)
            if self.mode!="infer":
                sampel_label_ids = sample[1][:max_seq_len] + [0 for _ in range(max_seq_len-sample_words_len)]
                batch_label_ids.append(sampel_label_ids)
            batch_lens.append(np.int64(sample_words_len))

        if self.mode == "infer":
            return batch_word_ids, batch_lens
        else:
            return batch_word_ids, batch_label_ids, batch_lens



class Reader():
    """data reader"""
    def __init__(self, args):
        # read dict
        self.word2id_dict = load_kv_dict(
            args.word_dict_path, reverse=True, value_func=np.int64)
        self.id2word_dict = load_kv_dict(args.word_dict_path)
        self.label2id_dict = load_kv_dict(
            args.label_dict_path, reverse=True, value_func=np.int64)
        self.id2label_dict = load_kv_dict(args.label_dict_path)
        self.word_replace_dict = load_kv_dict(args.word_rep_dict_path)
        self.args = args

    @property
    def vocab_size(self):
        """vocabuary size"""
        return max(self.word2id_dict.values()) + 1

    @property
    def num_labels(self):
        """num_labels"""
        return max(self.label2id_dict.values()) + 1

    def get_num_examples(self, filename):
        """num of line of file"""
        return sum(1 for line in io.open(filename, "r", encoding='utf8'))

    def word_to_ids(self, words):
        """convert word to word index"""
        word_ids = []
        for word in words:
            word = self.word_replace_dict.get(word, word)
            if word not in self.word2id_dict:
                word = "OOV"
            word_id = self.word2id_dict[word]
            word_ids.append(word_id)

        return word_ids

    def label_to_ids(self, labels):
        """convert label to label index"""
        label_ids = []
        for label in labels:
            if label not in self.label2id_dict:
                label = "O"
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def create_dataset(self, mode, max_seq_len=64):
        return Dataset(self.args, mode, max_seq_len=max_seq_len,_word_to_ids=self.word_to_ids,_label_to_ids=self.label_to_ids)