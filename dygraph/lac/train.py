# -*- coding: UTF-8 -*-
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

import os
import sys
import math
import time
import random
import argparse
import multiprocessing

import numpy as np
import paddle
import paddle.fluid as fluid



np.set_printoptions(threshold=np.inf)
import reader
import utils
from sequence_labeling import lex_net, Chunk_eval, ChunkEvaluator
#from eval import test_process

# the function to train model
def do_train(args):

    data_reader = reader.Reader(args)
    if args.use_cuda: 
        print("paddle.distributed.ParallelEnv().dev_id: ",paddle.distributed.ParallelEnv().dev_id)
        place = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id) \
        if args.use_data_parallel else paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()

    paddle.disable_static(place)
    if args.use_data_parallel:
        strategy = paddle.distributed.init_parallel_env()
    if args.enable_ce:
        paddle.framework.manual_seed(102)
        np.random.seed(102)
        random.seed(102)
    
    train_dataset = data_reader.create_dataset("train")
    if args.use_data_parallel:
        train_sampler = paddle.io.DistributedBatchSampler(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, drop_last=True)
    else:
        train_sampler = paddle.io.BatchSampler(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_sampler=train_sampler, places=place, return_list=True, collate_fn=train_dataset.padding_batch)

    test_dataset = data_reader.create_dataset("test")
    test_sampler = paddle.io.BatchSampler(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_sampler=test_sampler, places=place, return_list=True, collate_fn=test_dataset.padding_batch)
    
    model = lex_net(args, data_reader.vocab_size, data_reader.num_labels)
    if args.use_data_parallel:
        model = paddle.DataParallel(model, strategy)
    optimizer = paddle.optimizer.Adam(learning_rate=args.base_learning_rate, parameters=model.parameters())
    chunk_eval = Chunk_eval(int(math.ceil((data_reader.num_labels - 1) / 2.0)), "IOB")
    paddle.metric.chunk_eval
    num_train_examples = data_reader.get_num_examples(args.train_data)
    max_train_steps = args.epoch * num_train_examples // args.batch_size
    print("Num train examples: %d" % num_train_examples)
    print("Max train steps: %d" % max_train_steps)

    step = 0
    print_start_time = time.time()
    chunk_evaluator = ChunkEvaluator()
    chunk_evaluator.reset()

    def test_process(reader, chunk_evaluator):
        model.eval()
        chunk_evaluator.reset()

        start_time = time.time()
        for batch in reader():
            words, targets, length = batch
            try:
                crf_decode = model(words, length=length)
            except:
                # import pdb;pdb.set_trace()
                # crf_decode = model(words, length=length)
                print(words)
                print(length)
                raise
            (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
                num_correct_chunks) = chunk_eval(
                    input=crf_decode,
                    label=targets,
                    seq_length=length)
            chunk_evaluator.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())
        
        precision, recall, f1 = chunk_evaluator.eval()
        end_time = time.time()
        print("[test] P: %.5f, R: %.5f, F1: %.5f, elapsed time: %.3f s" %
            (precision, recall, f1, end_time - start_time))
        model.train()

    ce_time = []
    ce_infor = []
    for epoch_id in range(args.epoch):
        for batch in train_loader():
            words, targets, length = batch
            # print(words)
            # print(targets)
            # print(length)

            start_time = time.time()
            avg_cost, crf_decode = model(words, targets, length)
            if args.use_data_parallel:
                avg_cost = model.scale_loss(avg_cost)
                avg_cost.backward()
                model.apply_collective_grads()
            else:
                avg_cost.backward()
            optimizer.minimize(avg_cost)
            model.clear_gradients()
            end_time = time.time()

            if step % args.print_steps == 0:
                (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
                    num_correct_chunks) = chunk_eval(
                    input=crf_decode,
                    label=targets,
                    seq_length=length)
                outputs = [avg_cost, precision, recall, f1_score]
                avg_cost, precision, recall, f1_score = [np.mean(x.numpy()) for x in outputs]

                print("[train] step = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time %.5f" % (
                    step, avg_cost, precision, recall, f1_score, end_time - start_time))
                ce_time.append(end_time - start_time)
                ce_infor.append([precision, recall, f1_score])

            if step % args.validation_steps == 0:
                test_process(test_loader, chunk_evaluator)

            # save checkpoints
            if step % args.save_steps == 0 and step != 0:
                save_path = os.path.join(args.model_save_dir, "step_" + str(step))
                paddle.save(model.state_dict(), save_path)
            step += 1

    if args.enable_ce and paddle.distributed.ParallelEnv.local_rank == 0:
        card_num = paddle.framework.core.get_cuda_device_count
        _p = 0
        _r = 0
        _f1 = 0
        _time = 0
        try:
            _time = ce_time[-1]
            _p = ce_infor[-1][0]
            _r = ce_infor[-1][1]
            _f1 = ce_infor[-1][2]
        except:
            print("ce info error")
        print("kpis\ttrain_duration_card%s\t%s" % (card_num, _time))
        print("kpis\ttrain_p_card%s\t%f" % (card_num, _p))
        print("kpis\ttrain_r_card%s\t%f" % (card_num, _r))
        print("kpis\ttrain_f1_card%s\t%f" % (card_num, _f1))


if __name__ == "__main__":
    # 参数控制可以根据需求使用argparse，yaml或者json
    # 对NLP任务推荐使用PALM下定义的configure，可以统一argparse，yaml或者json格式的配置文件。

    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser, 'conf/args.yaml')

    args = parser.parse_args()

    print(args)

    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrapper = lp(do_train)
    lp_wrapper(args)
    lp.print_stats()
    # do_train(args)
