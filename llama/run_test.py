# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
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
import transformers

llama_qlora_adapter_model = "<path_to_adapter"
cache_path = "/hf-cache/"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_llama_mem_pipe():
    from llama_mem import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        llama_qlora_adapter_model,
        cache_dir=cache_path,
    )

    model.to('cuda:0')

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            llama_qlora_adapter_model,
            cache_dir=cache_path,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
    mem_token = "<landmark>"
    special_tokens_dict = dict()
    special_tokens_dict["additional_special_tokens"] = [mem_token]

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    from transformers import pipeline
    llama_mem_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=model.device)
    return llama_mem_pipe


llama_mem_pipe = make_llama_mem_pipe()

mem_id = llama_mem_pipe.tokenizer.convert_tokens_to_ids("<landmark>")
llama_mem_pipe.model.set_mem_id(mem_id)
llama_mem_pipe.model.set_mem_cache_args(max_seq_len=255, mem_freq=50, top_k=5, max_cache_size=None)


pipes = {"mem": llama_mem_pipe}

import torch

import os
import random
import re
import requests

def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix
    
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 2000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key
            


def test_model(prompt_text, pass_key, model_name):
    response = pipes[model_name](prompt_text,num_return_sequences=1, max_new_tokens=10)[0]["generated_text"][len(prompt_text):]
    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]
    
    return pass_key


n_values = [1000, 5000, 8000, 10000, 12000, 14000, 18000, 20000, 25000, 38000]
num_tests = 5
models = ["mem"]
accuracies = {x: [] for x in models}
individual_results = {x: [] for x in models}

for n in n_values:
    
    correct_count = {x: 0 for x in models}
    
    n_results = {x: [] for x in models}
    for i in range(num_tests):
        print(f"\nRunning test {i + 1}/{num_tests} for n = {n}...")
        prompt_text, pass_key = generate_prompt(n)
        
        
        
        for model_name in models:
            num_tokens = len(pipes[model_name].tokenizer.encode(prompt_text))

            print("Number of tokens in this prompt: ", num_tokens)
            model_output = test_model(prompt_text, pass_key, model_name)
            print(f"Expected number in the prompt: {pass_key}, {model_name} output: {model_output}")

            if pass_key == model_output:
                correct_count[model_name] += 1
                n_results[model_name].append(1)
                print("Success!")
            else:
                n_results[model_name].append(0)
                print("Fail.")
    
    for model in models:
        accuracy = (correct_count[model] / num_tests) * 100
        print(f"Accuracy {model} for n = {n}: {accuracy}%")
        accuracies[model].append(accuracy)
        individual_results[model].append(n_results)
