from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import transformers
import os
import argparse
import shutil
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--push_to_hub", action="store_true")

    return parser.parse_args()

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



def main():
    args = get_args()

    if args.device == 'auto':
        device_arg = { 'device_map': 'auto' }
    else:
        device_arg = { 'device_map': { "": args.device} }

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )

    print(f"Loading PEFT: {args.peft_model_path}")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    print(f"Running merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    mem_token = "<landmark>"
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = "[PAD]"
    special_tokens_dict["additional_special_tokens"] = [mem_token]

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model.save_pretrained(f"{args.output_dir}")
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir}")

    #Adding the remote code & updating the config. This is required to get landmark working
    landmark_config = "configuration_landmark_llama.py"
    landmark_model = "modelling_landmark_llama.py"
    
    shutil.copyfile(f"./remote_code/{landmark_config}", f"{args.output_dir}/{landmark_config}")
    shutil.copyfile(f"./remote_code/{landmark_model}", f"{args.output_dir}/{landmark_model}")

    with open(f"{args.output_dir}/config.json", "r") as config_file:
        config_dict = json.load(config_file)

    config_dict["auto_map"] = {
        "AutoConfig": "configuration_landmark_llama.LlamaConfig",
        "AutoModel": "modelling_landmark_llama.LlamaModel",
        "AutoModelForCausalLM": "modelling_landmark_llama.LlamaForCausalLM",
        "AutoModelForSequenceClassification": "modelling_landmark_llama.LlamaForSequenceClassification"
    }

    with open(f"{args.output_dir}/config.json", "w") as config_file:
        json.dump(config_dict, config_file, indent=4)

if __name__ == "__main__" :
    main()