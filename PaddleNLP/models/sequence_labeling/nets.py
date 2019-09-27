"""
The function lex_net(args) define the lexical analysis network structure
"""
import sys
import os
import math

import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

def lex_net(word, args, vocab_size, num_labels, for_infer = True, target=None):
    """
    define the lexical analysis network structure
    word: stores the input of the model
    for_infer: a boolean value, indicating if the model to be created is for training or predicting.

    return:
        for infer: return the prediction
        otherwise: return the prediction
    """
    word_emb_dim = args.word_emb_dim
    grnn_hidden_dim = args.grnn_hidden_dim
    emb_lr = args.emb_learning_rate if 'emb_learning_rate' in dir(args) else 1.0
    crf_lr = args.emb_learning_rate if 'crf_learning_rate' in dir(args) else 1.0
    bigru_num = args.bigru_num
    init_bound = 0.1
    IS_SPARSE = True

    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        # fluid.layers.Print(pre_gru, message="pre_gru", summarize=10)

        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        # fluid.layers.Print(gru, message="gru", summarize=10)

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        # fluid.layers.Print(pre_gru_r, message="pre_gru_r", summarize=10)

        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        # fluid.layers.Print(gru_r, message="gru_r", summarize=10)

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        # fluid.layers.Print(bi_merge, message="bi_merge", summarize=10)
        return bi_merge

    def _net_conf(word, target=None):
        """
        Configure the network
        """
        word_embedding = fluid.layers.embedding(
            input=word,
            size=[vocab_size, word_emb_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))
        word_embedding=fluid.layers.Print(word_embedding, message="word_embedding", summarize=10)

        input_feature = word_embedding
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output
        bigru_output=fluid.layers.Print(bigru_output, message="bigru_output", summarize=10)

        emission = fluid.layers.fc(
            size=num_labels,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        # crf_cost = fluid.layers.Print(emission, message="crf_cost", summarize=10)

        emission=fluid.layers.Print(emission,message="emission",summarize=10)

        if not for_infer:
            crf_cost = fluid.layers.linear_chain_crf(
                input=emission,
                label=target,
                param_attr=fluid.ParamAttr(
                    name='crfw',
                    learning_rate=crf_lr))
            # crf_cost = fluid.layers.Print(crf_cost, message="crf_cost",summarize=10)

            avg_cost = fluid.layers.mean(x=crf_cost)
            # fluid.layers.Print(avg_cost, message="avg_cost",summarize=10)

            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))

            # crf_decode=fluid.layers.Print(crf_decode, message="crf_decode",summarize=10)

            return avg_cost,crf_decode, crf_cost, emission, bigru_output, word_embedding

        else:
            size = emission.shape[1]
            fluid.layers.create_parameter(shape = [size + 2, size],
                                          dtype=emission.dtype,
                                          name='crfw')
            crf_decode = fluid.layers.crf_decoding(
                input=emission, param_attr=fluid.ParamAttr(name='crfw'))

        return crf_decode

    if for_infer:
        return _net_conf(word)

    else:
        # assert target != None, "target is necessary for training"
        return _net_conf(word, target)
