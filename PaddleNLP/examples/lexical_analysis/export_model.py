# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse

import numpy as np
import paddle
from paddlenlp.data import Pad, Tuple, Stack
from paddlenlp.metrics import ChunkEvaluator

from data import LacDataset, parse_lac_result
from model import BiGruCrf


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="The folder where the dataset is located.")
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=128,
        help="The dimension in which a word is embedded.")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="The number of hidden nodes in the GRU layer.")

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="Path to save the exported inference model.", )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    infer_dataset = LacDataset(args.data_dir, mode='infer')

    # build model and load trained parameters
    model = BiGruCrf(args.emb_dim, args.hidden_size, infer_dataset.vocab_size,
                     infer_dataset.num_labels)
    # model = paddle.Model(network)
    # model.prepare()
    weight_state_dict = paddle.load(args.model_path)
    model.set_state_dict(weight_state_dict)
    # model.eval()
    # switch to eval model
    # model.eval()
    # convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64"),  # word_ids
            paddle.static.InputSpec(
                shape=[None], dtype="int64")  # word_lens
        ])
    # save converted static graph model
    paddle.jit.save(model, args.output_path)


if __name__ == "__main__":
    main()
