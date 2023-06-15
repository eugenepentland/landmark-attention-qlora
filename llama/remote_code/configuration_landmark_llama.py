# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" LLaMA model configuration"""

from transformers import LlamaConfig as HFLlamaConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class LlamaConfig(HFLlamaConfig):
    model_type = "llama"

    def __init__(
        self,
        mem_id=32001,
        mem_freq=50,
        mem_top_k=5,
        mem_max_seq_len=255,
        mem_max_cache_size=None,
        **kwargs,
    ):
        self.mem_id = mem_id
        self.mem_freq = mem_freq
        self.mem_top_k = mem_top_k
        self.mem_max_seq_len = mem_max_seq_len
        self.mem_max_cache_size = mem_max_cache_size
        super().__init__(**kwargs)