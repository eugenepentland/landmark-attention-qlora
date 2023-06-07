# Landmark Attention QLoRA

This is an experiemental repo attempting to create a landmark attention model using QLoRA.

**This repo currently IS NOT complete and still a work in progress.**

Testing so far has been run on a linux system with a RTX 8000 48GB VRAM graphics card. We have been able to train a 3B and 7B model using 20GB and 29GB of VRAM respectivley. The output models have not been tested yet.

## Setup
1. Clone the repo
2. Create a Conda enviroment
3. Install the requirements.txt from /landmark-attention-qlora/llama
4. Train a model. The model name output dir, and cache will need to be updated to your local directories

Configuration notes:
If you are running on a newer cloud GPU, you will want to add --bfloat16 --tf32 True when you run the training for better performance.
```
/llama/run.sh
```
or
```
python3 train-qlora.py  
    --model_name_or_path /home/toast/models/wizardLM-7B-HF 
    --output_dir /home/toast/models/output  
    --cache_dir /home/toast/hf-cache/ 
    --num_train_epochs 1  
    --per_device_train_batch_size 1     
    --gradient_accumulation_steps 16     
    --learning_rate 2e-5     
    --weight_decay 0.1     
    --warmup_ratio 0.03     
    --lr_scheduler_type "cosine"     
    --logging_steps 1     
    --max_steps 1000 
```


## How it was made
The original landmark attention repo was taken and how the model gets loaded was replaced with how QLoRA loads a model. QLoRA uses AutoModelForCasualLM and that needed to be replaced with landmarks custom LlamaForCausalLM.

Currently gradient checking had to be disabled to get it to run, so if that can be figured out, there can be further memory useage improvements.
