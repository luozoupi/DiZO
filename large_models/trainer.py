# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
from collections import deque
import numpy as np

from tqdm.auto import tqdm
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from collections import deque

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    get_reporting_integration_callbacks,
    hp_params,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
    is_deepspeed_zero3_enabled,
    deepspeed_init,
)

from transformers.hyperparameter_search import default_hp_search_backend

# Define is_fairscale_available function since it's no longer in transformers.integrations
def is_fairscale_available():
    try:
        import fairscale
        return True
    except ImportError:
        return False

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
# from transformers.trainer_utils import (
#     PREFIX_CHECKPOINT_DIR,
#     BestRun,
#     EvalLoopOutput,
#     EvalPrediction,
#     FSDPOption,
#     HPSearchBackend,
#     HubStrategy,
#     IntervalStrategy,
#     PredictionOutput,
#     RemoveColumnsCollator,
#     ShardedDDPOption,
#     TrainerMemoryTracker,
#     TrainOutput,
#     default_compute_objective,
#     default_hp_space,
#     denumpify_detensorize,
#     enable_full_determinism,
#     find_executable_batch_size,
#     get_last_checkpoint,
#     has_length,
#     number_of_arguments,
#     seed_worker,
#     set_seed,
#     speed_metrics,
# )
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_torchdynamo_available,
    logging,
)
from transformers.pytorch_utils import is_torch_xla_available as is_torch_tpu_available
from transformers.trainer_callback import DefaultFlowCallback, ProgressCallback, TrainerState
from transformers.trainer_utils import has_length, TrainOutput, speed_metrics, HPSearchBackend
from transformers.utils.generic import ContextManagers
from scipy.special import loggamma

from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR

_is_native_cpu_amp_available = is_torch_greater_or_equal("1.10")

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets
import torch.optim as optim

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler
from utils import encode_prompt, Prediction

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    import optuna

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

from tasks import get_task
import torch.nn.functional as F
from metrics import calculate_metric
from collections import defaultdict


class RNG:
    def __init__(self, bits):
        self.seed = 0
        self.bits = bits
        self.idx = 0
        self.random_numbers = list(range(-2 ** (bits - 1), 2 ** (bits - 1)))
        random.shuffle(self.random_numbers)

    def step(self):
        number = self.random_numbers[self.idx]
        self.idx = (self.idx + 1) % len(self.random_numbers)
        return number


class RNGs:
    def __init__(self, bits, num_RNGs):
        self.bits = bits
        self.num_RNGs = num_RNGs
        self.rngs = [RNG(bits) for _ in range(num_RNGs)]
        self.idx = 0
        self.reverse_idx = 0
        self.get_norm()

    def get_norm(self):
        self.norm = {}
        for i in range(2 ** self.bits):
            rns = [rng.step() for rng in self.rngs]
            self.norm[rns[0]] = sum([rn ** 2 for rn in rns]) ** 0.5
        self.idx = 0

    def step(self):
        if self.idx % (2 ** self.bits - 1) == 0 and self.idx != 0:
            self.rngs.append(self.rngs.pop(0))

        self.idx += 1
        if (self.idx - 1) % (2 ** self.bits - 1) == 0 and self.idx != 1:
            self.reverse_idx = -((abs(self.reverse_idx) + 1) % self.num_RNGs)

        return [rng.step() for rng in self.rngs]

def direction_de(W):
    norm_W_c = torch.norm(W, dim=0, keepdim=True)

    V = W / norm_W_c

    m = norm_W_c

    return V, m




def DorefaW(w, bit, percent=0.01):

    scaling = True  # whether to use scaling factor, always true in our case
    if bit == 1:
        weight = w.detach()
        # sign = torch.sign(weight)
        scale = torch.max(torch.abs(weight))
        quantized = torch.sign(weight).float()
        residual = quantized - weight
        if scaling:
            w = (w + residual) * scale
        else:
            w = w + residual

    elif bit == 2:
        weight = w.detach()
        sign = torch.sign(weight)
        scale = torch.max(torch.abs(weight))
        quantized = (torch.abs(weight) > scale / 3).float()
        quantized *= sign
        residual = quantized - weight
        if scaling:
            w = (w + residual) * scale
        else:
            w = w + residual
    else:
        sign = torch.sign(w)
        scale = torch.max(torch.abs(w))
        w = torch.abs(w) / (scale + 1e-9)
        w = QuantizeW.apply(w, bit, 'fp')

        w = w * scale * sign

    return w

class QuantizeW(Function):
    @staticmethod
    def forward(ctx, input, bit, scheme='fp'):
        ctx.bit = bit
        # I. fix point:
        if scheme == 'fp':
            scale = float(2 ** bit - 1)
            out = torch.round(input * scale) / scale

        # II. power of 2:
        elif scheme == 'po2':
            out = 2 ** torch.round(torch.log2(input)) * (input > 2 ** (-2 ** bit + 1)).float()

        # III. sp2:
        elif scheme == 'sp2':
            size = input.size()
            y = input.reshape(-1)

            centroids = torch.tensor(
                [0, 2 ** -4, 2 ** -3, 2 ** -4 + 2 ** -3, 2 ** -2, 2 ** -2 + 2 ** -3, 2 ** -1, 2 ** -1 + 2 ** -3,
                 1]).cuda()
            mag = y - centroids.reshape(-1, 1)

            minimum = torch.min(torch.abs(mag), dim=0)[1]
            out = centroids[minimum]
            out = out.reshape(size)
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def backward(ctx, grad_output):
        bit = ctx.bit
        # bit = 8 - 1

        sign = torch.sign(grad_output)
        scaling = torch.max(torch.abs(grad_output))
        gradient = torch.abs(grad_output) / scaling

        scale = float(2 ** bit - 1)
        near = False
        if near:  # nearest rounding
            grad_input = torch.round(gradient * scale) / scale
        else:  # stochastic rounding
            """
            #randtensor = torch.rand(gradient.shape).to(gradient.device) - 0.5  # random tensor in [-0.5,0.5)  (too slow)
            randtensor = torch.cuda.FloatTensor(gradient.shape).uniform_(-0.5, 0.5).to(gradient.device) # random tensor in [-0.5,0.5)
            grad_input = torch.round((gradient * scale) + randtensor) / scale
            # e.g. gradient * scale = 3.3, add random value in [-0.5,0.5)
            # if random value is in   0.2~0.5 -> 4 (30%)
            # else                   -0.5~0.2 -> 3 (70%)
            """
            # random rounding
            randtensor = torch.cuda.FloatTensor(gradient.shape).uniform_(0, 1).to(gradient.device)
            randtensor = torch.round(randtensor) - 0.5  # only have 50% 0.5 and 50% -0.5
            grad_input = torch.round((gradient * scale) + randtensor) / scale
            # e.g. gradient * scale = 3.3, add random value (0.5 or -0.5)
            # if random value is 0.5, 3.3+0.5 = 3.8 -> 4 (50%)
            # else              -0.5, 3.3-0.5 = 2.8 -> 3 (50%)

        return grad_input * sign * scaling, None, None
# 
class DiZO(nn.Module):
    def __init__(self, model, norm_mode, exclude_list=[]) -> None:
        super().__init__()
        self.norm_mode = norm_mode
        self.exclude_list = exclude_list
        self.threshold = torch.nn.Hardtanh(0, 1)
        self.constraints_name = []
        self.constraints = []
        self.id_name_map = {}
        self.create_contraint(model)  # Create constraint place holders
        self.constraints = nn.ParameterList(self.constraints)
        self.constraints = self.constraints.to('cuda')
        self.init = True
        self.times = 0
        self.alpha = {}
        self.n = {}
        self.include_list = []

    def create_contraint(self, model):

        for name, para in model.named_parameters():
            if name not in self.exclude_list and para.requires_grad:
                self.constraints_name.append(name)
                temp = nn.Parameter(torch.Tensor([0]), requires_grad=True)
                self.constraints.append(temp)
                self.id_name_map[len(self.constraints_name) - 1] = name

    def apply_constraints(
            self,
            new,
            pre_trained,
            constraint_iterator,
            apply=False,
    ):

        self.include_list = []
        for (name, new_para), anchor_para in zip(
                new.named_parameters(), pre_trained.parameters()
        ):
            new_para.requires_grad = False
            if name not in self.exclude_list:
                alpha = self._project_ratio(
                    new_para,
                    anchor_para,
                    constraint_iterator,
                )

                # if apply:
                self.alpha[name] = alpha
                self.include_list.append(name)
                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()

                if apply:
                    with torch.no_grad():
                        new_para.copy_(temp)
                else:
                    new_para.requires_grad = False
                    new_para.copy_(temp)


    def reverse_constraints(self, new, pre_trained):
        for (name, new_para), anchor_para in zip(
                new.named_parameters(), pre_trained.parameters()
        ):
            new_para.requires_grad = False
            if name not in self.exclude_list:

                alpha = self.alpha[name]
                v = (new_para.detach() - anchor_para.detach()) / alpha
                temp = v + anchor_para.detach()

                with torch.no_grad():
                    new_para.copy_(temp)

    def _project_ratio(self, new, anchor, constraint_iterator):

        t = new.detach() - anchor.detach()

        if "l2" in self.norm_mode:
            norms = torch.norm(t)  # L2 norm
        else:
            norms = torch.sum(torch.abs(t), dim=tuple(range(1, t.dim())), keepdim=True)  # MARS norm

        constraint = next(constraint_iterator)

        ratio = constraint / (norms + 1e-8)
        return ratio

    def _clip(self, constraint, norms):
        return torch.nn.functional.hardtanh(constraint, 1e-8, norms.max())

    @staticmethod
    def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None,
                                     return_dict=None, **kwargs):
        """
        This is to replace the original forward function of Transformer models to enable:
        (1) Partial target sequence: loss will only be calculated on part of the sequence
        (2) Classification-style training: a classification loss (CE) will be calculated over several options
        Input:
        - input_ids, labels: same as the original forward function
        - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
          last option_len tokens
        - num_options: a list of int indicating the number of options for each example (this will be #label
          words for classification tasks and #choices for multiple choice tasks), and a classification loss
          will be calculated.
        """

        outputs = self.forward(input_ids=input_ids, **kwargs)

        if labels is None:
            return outputs
        logits = outputs.logits

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
        shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
        shift_labels[shift_labels == self.config.pad_token_id] = -100

        # Apply option len (do not calculate loss on the non-option part)
        if option_len is not None:
            for _i, _len in enumerate(option_len):
                shift_labels[_i, :-_len] = -100

        # Calculate the loss
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        if num_options is not None:
            # Train as a classification tasks
            log_probs = F.log_softmax(shift_logits, dim=-1)
            mask = shift_labels != -100  # Option part
            shift_labels[~mask] = 0  # So that it doesn't mess up with indexing

            selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(
                -1)  # (bsz x num_options, len)
            selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1)  # (bsz x num_options)

            if any([x != num_options[0] for x in num_options]):
                # Multi choice tasks with different number of options
                loss = 0
                start_id = 0
                count = 0
                while start_id < len(num_options):
                    end_id = start_id + num_options[start_id]
                    _logits = selected_log_probs[start_id:end_id].unsqueeze(0)  # (1, num_options)
                    _labels = labels[start_id:end_id][0].unsqueeze(0)  # (1)
                    loss = loss_fct(_logits, _labels) + loss
                    count += 1
                    start_id = end_id
                loss = loss / count
            else:
                num_options = num_options[0]
                selected_log_probs = selected_log_probs.view(-1, num_options)  # (bsz, num_options)
                labels = labels.view(-1, num_options)[:, 0]  # Labels repeat so we only take the first one
                loss = loss_fct(selected_log_probs, labels)
        else:
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
        )


    def perturb_gamma(self, delta, ts=None, zs=None, tau=None, zo_eps=None):
        if zs is None:
            pre = False
            zs = {}
        else:
            pre = True

        for i, (name, gamma) in enumerate(self.constraints.named_parameters()):
            if not pre:
                z = torch.normal(0, 1, size=(1,), device=gamma.device, dtype=gamma.dtype)
                z = torch.clip(z, (-tau / zo_eps) * ts[i], (tau / zo_eps) * ts[i])
                zs[name] = z
            else:
                z = zs[name]

            gamma.data = gamma.data + delta * z * zo_eps

        return zs

    def zo_forward(self, new=None, pre_trained=None, x=None, apply=False):

        if apply:
            constraint_iterator = iter(self.constraints)
            self.apply_constraints(
                new,
                pre_trained,
                constraint_iterator,
                apply=apply,
            )
            # print([each.item() for each in self.alpha.values()], np.mean([each.item() for each in self.ts]))
        else:

            ts = []

            with torch.no_grad():

                for (name, para), anchor in zip(new.named_parameters(), pre_trained.parameters()):
                    if name not in self.exclude_list:
                        norm = torch.norm(para.data - anchor.data)
                        ts.append(norm)

                if self.init:
                    for i, (name, gamma) in enumerate(self.constraints.named_parameters()):
                        gamma.data = ts[i]

                tau = 0.2
                zo_eps = 0.1
                step_size = 2

                zs = self.perturb_gamma(1, ts=ts, tau=tau, zo_eps=zo_eps)
                constraint_iterator = iter(self.constraints)
                self.apply_constraints(new, pre_trained, constraint_iterator)
                loss1 = self.forward_wrap_with_option_len(new, **x, return_dict=True).loss
                self.reverse_constraints(new, pre_trained)

                self.perturb_gamma(-2, ts=ts, zs=zs, tau=tau, zo_eps=zo_eps)
                constraint_iterator = iter(self.constraints)
                self.apply_constraints(new, pre_trained, constraint_iterator)
                loss2 = self.forward_wrap_with_option_len(new, **x, return_dict=True).loss
                self.reverse_constraints(new, pre_trained)

                self.perturb_gamma(1, ts=ts, zs=zs, tau=tau, zo_eps=zo_eps)
                grad = (loss1 - loss2) / (2 * zo_eps)

                for i, (name, gamma) in enumerate(self.constraints.named_parameters()):

                    tmp_z = zs[name]
                    gamma.data = torch.clip(gamma.data - step_size * ts[i] * grad * tmp_z, (1 - tau) * ts[i], (1 + tau) * ts[i])

            self.ts = ts

    # def zo_forward(self, new=None, pre_trained=None, x=None, apply=False):
    #
    #     if apply:
    #         constraint_iterator = iter(self.constraints)
    #         self.apply_constraints(
    #             new,
    #             pre_trained,
    #             constraint_iterator,
    #             apply=apply,
    #         )
    #     else:
    #         copy_model = copy.deepcopy(new)
    #         copy_model.eval()
    #         zs = {}
    #         ts = []
    #
    #         with torch.no_grad():
    #
    #             for (name, para), anchor in zip(copy_model.named_parameters(), pre_trained.parameters()):
    #                 if name not in self.exclude_list:
    #                     norm = torch.norm(para.data - anchor.data)
    #                     ts.append(norm)
    #
    #             # parameters_dict = {name: param.item() for name, param in self.constraints.named_parameters()}
    #             loss0 = self.forward_wrap_with_option_len(copy_model, **x, return_dict=True).loss
    #             # + 1e-3 * sum(p.abs().sum() for p in self.constraints.parameters())
    #
    #             scale = 0.1
    #             scale_v = 1
    #             scale_other = 0.3
    #             if self.init:
    #                 for i, (name, para) in enumerate(self.constraints.named_parameters()):
    #                     if 'self_attn.v_proj' in self.id_name_map[i]:
    #                         para.data = ts[i] * scale_v
    #                     else:
    #                         para.data = ts[i] * scale_other
    #                         # para.data = torch.normal(0, ts[i] * scale_other, size=(1,)).to(para.device)
    #                     zs[name] = 0
    #             else:
    #                 for i, (name, para) in enumerate(self.constraints.named_parameters()):
    #                     if 'self_attn.v_proj' in self.id_name_map[i]:
    #                         z = torch.normal(0, ts[i] * scale_v, size=(1,)).to(para.device)
    #                         para.data = para.data + scale * z
    #                     else:
    #                         z = torch.normal(0, ts[i] * scale_other, size=(1,)).to(para.device)
    #                         # para.data = para.data + scale * z
    #                         para.data = torch.clip(para.data + scale * z, -scale_other * ts[i], scale_other * ts[i])
    #                     zs[name] = z.item()
    #
    #             constraint_iterator = iter(self.constraints)
    #             self.apply_constraints(copy_model, pre_trained, constraint_iterator)
    #             loss1 = self.forward_wrap_with_option_len(copy_model, **x, return_dict=True).loss
    #             # + 1e-3 * sum(p.abs().sum() for p in self.constraints.parameters())
    #             # + 5e-4 * sum(p.abs().sum() for p in self.constraints.parameters())
    #
    #
    #             if loss0 < loss1:
    #                 for i, (name, para) in enumerate(self.constraints.named_parameters()):
    #                     para.data = para.data - 2 * scale * zs[name]

    @staticmethod
    def initialize_population(num_layers, population_size):
        population = torch.empty((population_size, num_layers)).uniform_(-0.1, 0.1)
        return population

    def apply_projections(self, copy_model, pre_trained, projections):

        i = 0
        delta = []
        for (name, new_para), anchor_para in zip(copy_model.named_parameters(), pre_trained.parameters()):
            if name not in self.exclude_list:
                t = new_para.detach() - anchor_para.detach()

                delta.append(t)

                norms = torch.norm(t)

                alpha = projections[i] / (norms + 1e-8)

                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()
                new_para.data = temp
                i += 1

        return copy_model, delta

    def reverse_projections(self, copy_model, pre_trained, delta):

        i = 0
        for (name, new_para), anchor_para in zip(copy_model.named_parameters(), pre_trained.parameters()):
            if name not in self.exclude_list:
                t = delta[i]

                new_para.data = anchor_para.data + t
                i += 1

        return copy_model

    def forward(
            self,
            new=None,
            pre_trained=None,
            x=None,
            apply=False,
    ):
        constraint_iterator = iter(self.constraints)

        if apply:
            self.apply_constraints(
                new,
                pre_trained,
                constraint_iterator,
                apply=apply,
            )
        else:

            # print(torch.cuda.memory_allocated() // 1024 // 1024)
            copy_model = copy.deepcopy(new)
            copy_model.eval()

            self.apply_constraints(copy_model, pre_trained, constraint_iterator)

            out = self.forward_wrap_with_option_len(copy_model, **x, return_dict=True)
            pgm_loss = out.loss + 1e-3 * sum(p.abs().sum() for p in self.constraints.parameters())

            return pgm_loss

# 
class dizo_trainer():
    def __init__(
            self,
            base_model,
            pgmloader,
            norm_mode,
            proj_lr,
            max_iters,
            exclude_list=[]
    ) -> None:
        # super().__init__(model=None, model_init=None)
        self.device = torch.device("cuda")
        self.proj_lr = proj_lr
        self.max_iters = max_iters
        self.exclude_list = exclude_list
        self.dizo = DiZO(base_model, norm_mode=norm_mode, exclude_list=exclude_list).to(base_model.device)
        self.pre_trained = base_model
        self.pgm_optimizer = torch.optim.Adam(self.dizo.parameters(), lr=self.proj_lr)
        self.pgmloader = pgmloader
        self.dataset_iterator = iter(self.pgmloader)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.i = 0

    def dizo_bayesian_search(self, model, base_model, apply=False):
        if not apply:

            try:
                data = next(self.dataset_iterator)
            except StopIteration:
                self.dataset_iterator = iter(self.pgmloader)
                data = next(self.dataset_iterator)

            for each in data:
                data[each] = data[each].to(self.device)
            # data = self._prepare_inputs(data)

        self.dizo(model, self.pre_trained, apply=True)

    def dizo_zo_iters(self, model, base_model, apply=False):
        if not apply:
            self.count = 0
            self.dizo = self.dizo.to(self.device)
            self.dizo.init = True
            while self.count < self.max_iters:

                try:
                    data = next(self.dataset_iterator)
                except StopIteration:
                    self.dataset_iterator = iter(self.pgmloader)
                    data = next(self.dataset_iterator)

                for each in data:
                    data[each] = data[each].to(self.device)

                self.dizo.zo_forward(model, base_model, x=data)
                self.dizo.init = False
                self.count += 1

        self.dizo.zo_forward(model, self.pre_trained, apply=True)
        self.i += 1

    def dizo_iters(self, model, base_model, apply=False):
        if not apply:
            self.count = 0
            self.dizo = self.dizo.to(self.device)

            while self.count < self.max_iters:

                try:
                    data = next(self.dataset_iterator)
                except StopIteration:
                    self.dataset_iterator = iter(self.pgmloader)
                    data = next(self.dataset_iterator)

                for each in data:
                    data[each] = data[each].to(self.device)
                # data = self._prepare_inputs(data)

                pgm_loss = self.dizo(model, base_model, x=data)
                self.pgm_optimizer.zero_grad()
                pgm_loss.backward()
                self.pgm_optimizer.step()

                self.count += 1

        self.dizo(model, self.pre_trained, apply=True)
        self.i += 1


class OurTrainer(Trainer):
    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    @property
    def tokenizer(self):
        """Backward compatibility property for accessing tokenizer via processing_class"""
        return getattr(self, 'processing_class', None)

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """

        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        self.task = get_task(self.args.task_name)
        self.objective = 0
        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                # Enhanced model support for token prediction layer
                if model.config.model_type == "opt":
                    return model.lm_head
                elif model.config.model_type in ["gpt2", "gpt"]:
                    return model.lm_head
                elif model.config.model_type in ["llama", "llama2", "llama3"]:
                    return model.lm_head
                elif model.config.model_type in ["qwen", "qwen2"]:
                    return model.lm_head
                elif model.config.model_type in ["mistral", "mixtral"]:
                    return model.lm_head
                elif model.config.model_type == "falcon":
                    return model.lm_head
                elif model.config.model_type in ["roberta", "bert"]:
                    # For classification models
                    return model.classifier if hasattr(model, 'classifier') else model.lm_head
                else:
                    # Default fallback
                    if hasattr(model, 'lm_head'):
                        return model.lm_head
                    elif hasattr(model, 'classifier'):
                        return model.classifier
                    else:
                        raise NotImplementedError(f"Model type {model.config.model_type} not supported for linear probing")

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]

                    # Shift the target (bc it's autoregressive LM) and add the corresponding part
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Enhanced bias handling for different model types
            if self.model.config.model_type in ["opt", "gpt2", "gpt"]:
                use_bias = False
            elif self.model.config.model_type in ["llama", "llama2", "llama3", "mistral", "mixtral"]:
                use_bias = False  # Llama-style models typically don't use bias
            elif self.model.config.model_type in ["qwen", "qwen2"]:
                use_bias = False  # Qwen models typically don't use bias
            elif self.model.config.model_type == "falcon":
                use_bias = False  # Falcon models typically don't use bias
            elif self.model.config.model_type in ["roberta", "bert"]:
                use_bias = True   # BERT-style models use bias
            else:
                use_bias = False  # Default to no bias for unknown models
                logger.warning(f"Unknown model type {self.model.config.model_type}, defaulting to use_bias=False")
            # Set early stopping
            tol = 0.01 if self.args.lp_early_stopping else 1e-4  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                hasattr(self.args, 'sharded_ddp') and self.args.sharded_ddp is not None
                and self.args.sharded_ddp != "simple"
                or is_sagemaker_mp_enabled()
                or hasattr(self, 'fsdp') and self.fsdp is not None
        )

        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        # self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        if args.trainer == 'zo':
            lr_lambda = lambda step: 1 - self.state.global_step / (2 * args.max_steps)
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step


        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if not is_torch_greater_or_equal("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # : exclude the layers do not need projection
        if self.named_parameters_to_optim[0][0] != 'model.decoder.embed_tokens.weight':
            self.exclude_list = [each for each in list(self.model.state_dict().keys()) if 'lora' not in each]
        else:
            self.exclude_list = ['model.decoder.embed_tokens.weight'] + [name for name, _ in
                                                                         self.named_parameters_to_optim if
                                                                         'self_attn.v_proj.weight' not in name and 'self_attn.k_proj.weight' not in name]
            self.named_parameters_to_optim = self.named_parameters_to_optim[1:]

        # : remove the unnecessary parameters to cpu for memory saving
        if args.enhanced in ['zo', 'fo']:
            self.base_model = copy.deepcopy(self.model)
            for name, param in self.base_model.named_parameters():
                if name in self.exclude_list:
                    param.data = param.data.to('cpu')
            self.dizo_trainer = dizo_trainer(self.base_model, train_dataloader, 'l2norm', 0.1, 10, self.exclude_list)

        else:
            args.enhanced = None

        self.loss_list = []
        self.random_vector = {}

        self.accuracy = []

        for epoch in range(epochs_trained, num_train_epochs):

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)


            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                if args.trainer == "zo":
                    tr_loss_step = self.zo_step(model, inputs)

                else:
                    if (
                            ((step + 1) % args.gradient_accumulation_steps != 0)
                            and args.local_rank != -1
                            and args._no_sync_in_gradient_accumulation
                    ):
                        # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                        with model.no_sync():
                            tr_loss_step = self.forward_wrap_with_option_len(model, **inputs, return_dict=True).loss
                            # tr_loss_step = self.training_step(model, inputs)
                    else:
                        inputs = self._prepare_inputs(inputs)
                        tr_loss_step = self.forward_wrap_with_option_len(model, **inputs, return_dict=True).loss
                        # tr_loss_step = self.training_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    if args.trainer == "zo":
                        self.zo_update(args, model)
                    else:
                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                            # deepspeed does its own clipping

                            if self.do_grad_scaling:
                                # Reduce gradients first for XLA
                                if is_torch_tpu_available():
                                    gradients = xm._fetch_gradients(self.optimizer)
                                    xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                                # AMP: gradients need unscaling
                                self.scaler.unscale_(self.optimizer)

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        optimizer_was_run = True
                        if self.deepspeed:
                            pass  # called outside the loop
                        elif is_torch_tpu_available():
                            if self.do_grad_scaling:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                xm.optimizer_step(self.optimizer)
                        elif self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run and not self.deepspeed:
                            self.lr_scheduler.step()
                        model.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    log_step = 50 if args.trainer == 'zo' else 10
                    if self.state.global_step % log_step == 0:
                        logger.info(
                            {'loss': round(tr_loss_step.item(), 4), 'epoch': epoch, 'lr': self._get_learning_rate()})

                    self.loss_list.append(tr_loss_step.item())
                    self.args.eval_steps = 100

                    if self.state.global_step % self.args.eval_steps == 0:

                        path = 'loss_acc/{}_{}_{}_enhanced_{}'.format(self.args.task_name, args.trainer, 'lora' if args.lora else 'ft', args.enhanced)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        np.save(path + '/' + 'loss_list_seed_{}.npy'.format(args.seed), self.loss_list)
                        predictions = []
                        for eval_sample in self.eval_dataset:
                            predictions.append(
                                self.one_step_pred([], eval_sample, verbose=False)
                            )
                        metric_name = getattr(self.task, "metric_name", "accuracy")
                        metrics = {metric_name: calculate_metric(predictions, metric_name)}
                        metrics["global_step"] = self.state.global_step
                        logger.info(f"Eval results: {metrics}")
                        self.accuracy.append(metrics['accuracy'])
                        np.save(path + '/' + 'accuracy_seed_{}.npy'.format(args.seed), self.accuracy)

                        if metrics['accuracy'] >= self.objective:
                            logger.info("Best dev result: {}".format(metrics['accuracy']))
                            self.objective = metrics['accuracy']
                            # self.save_model(self.args.output_dir)

                            # Now we save this to (CPU) memory instead of disk <-- much faster
                            self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps
            if self.zo_random_seed is not None:
                self.random_vector[name] = z

    @staticmethod
    def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None,
                                     return_dict=None, **kwargs):
        """
        This is to replace the original forward function of Transformer models to enable:
        (1) Partial target sequence: loss will only be calculated on part of the sequence
        (2) Classification-style training: a classification loss (CE) will be calculated over several options
        Input:
        - input_ids, labels: same as the original forward function
        - option_len: a list of int indicating the option lengths, and loss will be calculated only on the
          last option_len tokens
        - num_options: a list of int indicating the number of options for each example (this will be #label
          words for classification tasks and #choices for multiple choice tasks), and a classification loss
          will be calculated.
        """
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, **kwargs)

        if labels is None:
            return outputs
        logits = outputs.logits

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        # Here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
        shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
        shift_labels[shift_labels == self.config.pad_token_id] = -100

        # Apply option len (do not calculate loss on the non-option part)
        if option_len is not None:
            for _i, _len in enumerate(option_len):
                shift_labels[_i, :-_len] = -100

        # Calculate the loss
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        if num_options is not None:
            # Train as a classification tasks
            log_probs = F.log_softmax(shift_logits, dim=-1)
            mask = shift_labels != -100  # Option part
            shift_labels[~mask] = 0  # So that it doesn't mess up with indexing

            selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(
                -1)  # (bsz x num_options, len)
            selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1)  # (bsz x num_options)

            if any([x != num_options[0] for x in num_options]):
                # Multi choice tasks with different number of options
                loss = 0
                start_id = 0
                count = 0
                while start_id < len(num_options):
                    end_id = start_id + num_options[start_id]
                    _logits = selected_log_probs[start_id:end_id].unsqueeze(0)  # (1, num_options)
                    _labels = labels[start_id:end_id][0].unsqueeze(0)  # (1)
                    loss = loss_fct(_logits, _labels) + loss
                    count += 1
                    start_id = end_id
                loss = loss / count
            else:
                num_options = num_options[0]
                selected_log_probs = selected_log_probs.view(-1, num_options)  # (bsz, num_options)
                labels = labels.view(-1, num_options)[:, 0]  # Labels repeat so we only take the first one
                loss = loss_fct(selected_log_probs, labels)
        else:
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        if self.args.non_diff:
            # Non-differentiable objective (may require autoregressive generation)
            return self.zo_forward_nondiff(model, inputs)

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                with torch.no_grad():
                    # loss = self.compute_loss(model, inputs)
                    loss = self.forward_wrap_with_option_len(model, **inputs, return_dict=True).loss
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature,
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                              self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(
                    self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]

        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)

        return loss1

    def zo_update(self, args, model):
        """
        Update the parameters with the estimated gradients.
        """

        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

        # 
        if (self.state.global_step + 1) % 50 == 0 and args.enhanced:
            if args.enhanced == 'zo':
                self.dizo_trainer.dizo_zo_iters(model, base_model=self.base_model)
            else:
                self.dizo_trainer.dizo_iters(model, base_model=self.base_model)

        self.lr_scheduler.step()

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                hasattr(self.args, 'sharded_ddp') and self.args.sharded_ddp is not None
                and ("zero_dp_2" in str(self.args.sharded_ddp) or "zero_dp_3" in str(self.args.sharded_ddp))
                or hasattr(self, 'fsdp') and self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        with torch.inference_mode():
            self.model.eval()
            logits = self.model(input_ids=input_ids).logits
        labels = input_ids[0, 1:]
        logits = logits[0, :-1]
        log_probs = F.log_softmax(logits, dim=-1)

        selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
        selected_log_probs = selected_log_probs.cpu().detach()
        # Only return the option (candidate) part
        return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")

        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        outputs = []

        # For classification/multiple-choice, calculate the probabilities of all candidates
        for candidate_id, encoded_candidate in enumerate(encoded_candidates):
            selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
            if verbose:
                if candidate_id == 0:
                    logger.info("=== Candidate %d ===" % candidate_id)
                    logger.info(self.tokenizer.decode(encoded_candidate))
                else:
                    logger.info("=== Candidate %d (without context)===" % candidate_id)
                    logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

            outputs.append({"log_probs": selected_log_probs,
                            "sfc_log_probs": None})

        scores = [x['log_probs'].mean().item() for x in outputs]

        if verbose:
            logger.info(f"Prediction scores: {scores}")

        if isinstance(eval_sample.correct_candidate, list):
            # For some datasets there are multiple correct answers
            correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
        else:
            correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

        return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


