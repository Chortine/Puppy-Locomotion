# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os, sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)
print(__file__)
import isaacgym
from legged_gym.envs import *
from envs import *
from legged_gym.utils import get_args
from rl_utils.task_registry import task_registry
import torch
current_root = os.path.dirname(os.path.dirname(__file__))

def train(args):
    args.task = "wavego_flat"
    # args.task = 'a1_flat_default'
    # args.task = "a1"
    # args.task = "a1_flat"
    args.num_envs = 12000
    args.seed = 42
    args.resume = False
    # args.checkpoint = -1
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, log_root=os.path.join(current_root, 'logs/'))
    # ckpt_path = '/home/tianchu/Documents/code_qy/puppy-gym/logs/Oct09_16-21-21_run1/model_100.pt'
    # ppo_runner.load(ckpt_path)
    ckpt_path = os.path.join(current_root, 'logs/Nov22_18-55-42_run1/model_3690.pt')
    # ckpt_path = '/home/tianchu/Documents/code_qy/puppy-gym/logs/model_3690.pt'
    ppo_runner.load_for_rma_phase2(ckpt_path)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
