#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
import math
import numpy as np
import six

class padding_edit_distance(fluid.evaluator.EditDistance):
    def __init__(self, input, label, input_length, label_length, ignored_tokens=None, **kwargs):
        super(fluid.evaluator.EditDistance, self).__init__("edit_distance", **kwargs)
        main_program = self.helper.main_program
        if main_program.current_block().idx != 0:
            raise ValueError("You can only invoke Evaluator in root block")

        self.total_distance = self._create_state(
            dtype='float32', shape=[1], suffix='total_distance')
        self.seq_num = self._create_state(
            dtype='int64', shape=[1], suffix='seq_num')
        self.instance_error = self._create_state(
            dtype='int64', shape=[1], suffix='instance_error')

        # fluid.layers.Print(input_length,summarize=5)
        # fluid.layers.Print(label_length,summarize=5)
        distances, seq_num = fluid.layers.edit_distance(
            input=input, label=label, ignored_tokens=ignored_tokens, input_length=input_length, label_length=label_length)

        zero = fluid.layers.fill_constant(shape=[1], value=0.0, dtype='float32')
        compare_result = fluid.layers.equal(distances, zero)
        compare_result_int = fluid.layers.cast(x=compare_result, dtype='int64')
        seq_right_count = fluid.layers.reduce_sum(compare_result_int)
        instance_error_count = fluid.layers.elementwise_sub(
            x=seq_num, y=seq_right_count)
        total_distance = fluid.layers.reduce_sum(distances)
        fluid.layers.sums(
            input=[self.total_distance, total_distance],
            out=self.total_distance)
        fluid.layers.sums(input=[self.seq_num, seq_num], out=self.seq_num)
        fluid.layers.sums(
            input=[self.instance_error, instance_error_count],
            out=self.instance_error)
        self.metrics.append(total_distance)
        self.metrics.append(instance_error_count)


def conv_bn_pool(input,
                 group,
                 out_ch,
                 act="relu",
                 param=None,
                 bias=None,
                 param_0=None,
                 is_test=False,
                 pooling=True,
                 use_cudnn=False):
    tmp = input
    for i in six.moves.xrange(group):
        tmp = fluid.layers.conv2d(
            input=tmp,
            num_filters=out_ch[i],
            filter_size=3,
            padding=1,
            param_attr=param if param_0 is None else param_0,
            act=None,  # LinearActivation
            use_cudnn=use_cudnn)
        tmp = fluid.layers.batch_norm(
            input=tmp,
            act=act,
            param_attr=param,
            bias_attr=bias,
            is_test=is_test)
    if pooling:
        tmp = fluid.layers.pool2d(
            input=tmp,
            pool_size=2,
            pool_type='max',
            pool_stride=2,
            use_cudnn=use_cudnn,
            ceil_mode=True)

    return tmp


def ocr_convs(input,
              regularizer=None,
              gradient_clip=None,
              is_test=False,
              use_cudnn=False):
    b = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.0))
    w0 = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.0005))
    w1 = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.01))
    tmp = input
    tmp = conv_bn_pool(
        tmp,
        2, [16, 16],
        param=w1,
        bias=b,
        param_0=w0,
        is_test=is_test,
        use_cudnn=use_cudnn)

    tmp = conv_bn_pool(
        tmp,
        2, [32, 32],
        param=w1,
        bias=b,
        is_test=is_test,
        use_cudnn=use_cudnn)
    tmp = conv_bn_pool(
        tmp,
        2, [64, 64],
        param=w1,
        bias=b,
        is_test=is_test,
        use_cudnn=use_cudnn)
    tmp = conv_bn_pool(
        tmp,
        2, [128, 128],
        param=w1,
        bias=b,
        is_test=is_test,
        pooling=False,
        use_cudnn=use_cudnn)
    return tmp


def encoder_net(images,
                num_classes,
                seq_length,
                rnn_hidden_size=200,
                regularizer=None,
                gradient_clip=None,
                is_test=False,
                use_cudnn=False,
                ):
    conv_features = ocr_convs(
        images,
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        is_test=is_test,
        use_cudnn=use_cudnn)
    _, _, H, W = conv_features.shape
    sliced_feature = fluid.layers.im2sequence(
        input=conv_features,
        stride=[1, 1],
        filter_size=[H, 1])
    # -1 768

    reshape_sliced_feature=fluid.layers.reshape(sliced_feature,shape=[-1, H*W, sliced_feature.shape[-1]])
    #-1 384 768
    # print(reshape_sliced_feature)

    para_attr = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.02))
    bias_attr = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.02),
        learning_rate=2.0)
    bias_attr_nobias = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.02))

    fc_1 = fluid.layers.fc(input=reshape_sliced_feature,
                           size=rnn_hidden_size * 3,
                           param_attr=para_attr,
                           bias_attr=bias_attr_nobias,
                           num_flatten_dims=2)
    #-1 384 600
    # print(fc_1)

    fc_2 = fluid.layers.fc(input=reshape_sliced_feature,
                           size=rnn_hidden_size * 3,
                           param_attr=para_attr,
                           bias_attr=bias_attr_nobias,
                           num_flatten_dims=2)
    # print(fc_2)
    #-1 384 600

    gru_cell = fluid.layers.rnn.GRUCell(hidden_size=rnn_hidden_size, param_attr=para_attr,bias_attr=bias_attr,activation=fluid.layers.relu)

    gru_forward, _ = fluid.layers.rnn.rnn(cell=gru_cell, inputs=fc_1, sequence_length=seq_length)
    # print(gru_forward)
    # -1 384 200

    gru_backward, _ = fluid.layers.rnn.rnn(cell=gru_cell, inputs=fc_2, sequence_length=seq_length,is_reverse=True)
    # print(gru_backward)
    # -1 384 200

    w_attr = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.02))
    b_attr = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.0))

    fc_out = fluid.layers.fc(input=[gru_forward, gru_backward],
                             size=num_classes + 1,
                             param_attr=w_attr,
                             bias_attr=b_attr,
                             num_flatten_dims=2)
    # print(fc_out)
    # -1 384 96

    return fc_out


def ctc_train_net(args, data_shape, num_classes):
    L2_RATE = args.l2decay
    LR = args.lr
    MOMENTUM = args.momentum
    learning_rate_decay = None
    regularizer = fluid.regularizer.L2Decay(L2_RATE)

    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32',lod_level=0)
    label = fluid.layers.data(
        name='label', shape=[-1,1], dtype='int32', lod_level=0)
    label_length = fluid.layers.data(
        name='label_length', shape=[-1], dtype='int32', lod_level=0)
    seq_length=fluid.layers.data(
        name='seq_length', shape=[-1], dtype='int32', lod_level=0)

    fc_out = encoder_net(
        images,
        num_classes,
        seq_length,
        regularizer=regularizer,
        use_cudnn=True if args.use_gpu else False,
    )
    # print("fc_out",fc_out)
    # -1 384 96

    fc_out_t=fluid.layers.transpose(fc_out, perm=[1,0,2])
    # print("fc_out_t",fc_out_t)
    # 384 -1 96
    cost = fluid.layers.warpctc(
        input=fc_out, label=label, blank=num_classes, norm_by_times=True,input_length=seq_length,label_length=label_length)
    print("cost",cost)
    # 384 1

    sum_cost = fluid.layers.reduce_sum(cost)
    decoded_out, decoded_len = fluid.layers.ctc_greedy_decoder(
        input=fc_out, blank=num_classes,input_length=label_length)
    # print("decoded_out",decoded_out)
    # -1 384

    casted_label = fluid.layers.cast(x=label, dtype='int64')

    error_evaluator = padding_edit_distance(
        input=decoded_out, label=casted_label, input_length=decoded_len, label_length=label_length)

    inference_program = fluid.default_main_program().clone(for_test=True)
    if learning_rate_decay == "piecewise_decay":
        learning_rate = fluid.layers.piecewise_decay([
            args.total_step // 4, args.total_step // 2, args.total_step * 3 // 4
        ], [LR, LR * 0.1, LR * 0.01, LR * 0.001])
    else:
        learning_rate = LR

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate, momentum=MOMENTUM)
    _, params_grads = optimizer.minimize(sum_cost)
    model_average = None
    if args.average_window > 0:
        model_average = fluid.optimizer.ModelAverage(
            args.average_window,
            min_average_window=args.min_average_window,
            max_average_window=args.max_average_window)
    return sum_cost, error_evaluator, inference_program, model_average


def ctc_infer(images, num_classes, length, use_cudnn=True):
    fc_out = encoder_net(images, num_classes, length, is_test=True, use_cudnn=use_cudnn)
    return fluid.layers.ctc_greedy_decoder(input=fc_out, blank=num_classes,input_length=length)


def ctc_eval(data_shape, num_classes, use_cudnn=True):
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[1], dtype='int32', lod_level=1)
    length = fluid.layers.data(
        name='length', shape=[-1], dtype='int32', lod_level=0)
    img_length = fluid.layers.data(
        name='img_length', shape=[-1], dtype='int32', lod_level=0)

    fc_out = encoder_net(images, num_classes, length, is_test=True, use_cudnn=use_cudnn)
    decoded_out, decoded_len = fluid.layers.ctc_greedy_decoder(
        input=fc_out, blank=num_classes, input_length=length)

    casted_label = fluid.layers.cast(x=label, dtype='int64')
    error_evaluator = padding_edit_distance(
        input=decoded_out, label=casted_label, input_length=decoded_len, label_length=length)

    cost = fluid.layers.warpctc(
        input=fc_out, label=label, blank=num_classes, norm_by_times=True, input_length=img_length,label_length=length)

    return error_evaluator, cost
