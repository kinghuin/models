# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import math
import numpy
import warnings
from paddle import Tensor
from paddle.optimizer.lr import LRScheduler


class CosineAnnealingWithWarmupDecay(LRScheduler):
    def __init__(self,
                 max_lr,
                 min_lr,
                 warmup_step,
                 decay_steps,
                 last_epoch=-1,
                 verbose=False):

        self.decay_steps = decay_steps 
        self.warmup_step = warmup_step
        self.max_lr = max_lr 
        self.min_lr = min_lr
        print(self.decay_steps, self.warmup_step, self.max_lr, self.min_lr)
        super(CosineAnnealingWithWarmupDecay, self).__init__(
            max_lr, last_epoch, verbose)

    def get_lr(self):
        if self.warmup_step > 0 and self.last_epoch <= self.warmup_step:
            return float(self.max_lr) * (self.last_epoch) / self.warmup_step

        if self.last_epoch > self.decay_steps:
            return self.min_lr

        num_steps_ = self.last_epoch - self.warmup_step
        decay_steps_ = self.decay_steps - self.warmup_step
        decay_ratio = float(num_steps_) / float(decay_steps_)
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        return self.min_lr + coeff * (self.max_lr - self.min_lr) 
