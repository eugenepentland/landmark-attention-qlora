# Landmark Attention QLoRA

This is an experiemental repo attempting to create a landmark attention model using QLoRA.

**The concept has been validated, but further training is still required. The models do not work in oobooga yet. Evaluation must be run in the run_test.py file until the remote code has been written to allow it work in oobooga.**

A 7B model has been trained using 29GB of VRAM, for 500 steps/3 hours. It was functional all the way up to 7k context, but the accuracy was only about 60%. Further training should improve these results. Evaluation is still in progress and new models are currently being trained.

Testing so far has been run on a linux system with a RTX Quadro 8000 48GB VRAM graphics card.

## Setup
1. Clone the repo: git clone https://github.com/eugenepentland/landmark-attention-qlora.git
2. Create a Conda enviroment: conda create -n landmark 
3. Install the requirements.txt: cd /landmark-attention-qlora/llama; pip install -r requirements.txt
4. Train a model using train_qlora.py
5. Merge with the original weights using merge_peft.py
6. Evaluate using run_test.py


## Training
Configuration notes:
If you are running on a newer cloud GPU, you will want to add --bf16 True --tf32 True when you run the training for better performance.

```
python3 train_qlora.py  
    --model_name_or_path <path_to_llama_base_model> 
    --output_dir <output_directory> 
    --cache_dir <cache_directory> 
    --per_device_train_batch_size 2     
    --gradient_accumulation_steps 16     
    --learning_rate 2e-5     
    --weight_decay 0.1     
    --warmup_ratio 0.03     
    --lr_scheduler_type "cosine"     
    --logging_steps 1     
    --max_steps 10000 
    --bf16 False 
    --tf32 False 
```
## Merging
Merging typically provides faster inference times than using the QLoRA seperatley.
```
python3 merge_peft.py   
    --base_model_name_or_path <path_to_llama_model> 
    --peft_model_path <path_to_QLoRA_adapter> 
    --output_dir <output_path> 
```
## How it was made
The original landmark attention repo was taken and how the model gets loaded was replaced with how QLoRA loads a model. QLoRA uses AutoModelForCasualLM and that needed to be replaced with landmarks custom LlamaForCausalLM.
