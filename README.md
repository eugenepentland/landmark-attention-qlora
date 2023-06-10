# Landmark Attention QLoRA

Landmark Attention enables a 50x compression of an LLM's context into landmarks, making the process of selecting relevant tokens for answers more efficient, and allowing 2-16x longer context use without memory constraints. By integrating it with QLoRA, computational and training demands significantly decreased. The models were effectively trained on an H100 graphics card within 2-3 hours, at a total cost of $15.

![image](https://github.com/eugenepentland/landmark-attention-qlora/assets/32426720/50d36dae-3fd2-405f-9dc4-75d215f44903)

The following test was a needle in a haystack problem. The model is told that it will be provided a block of text, and it needs to find the pass key and tell the user what it is (llama/run_test.py). The results show that our 13B Minataur Landmark model gets comperable performance as the fully fine tuned 7B llama model.

## Models
Minotaur-13 & WizardLM-7B. Larger models will be released within the next few days.
https://huggingface.co/eugenepentland

Models ending with -Landmark-QLoRA are the adapter models only. They must be merged with the base model to be used. (/llama/merge_peft.py)

## Setup
1. Clone the repo: git clone https://github.com/eugenepentland/landmark-attention-qlora.git
2. Create a Conda enviroment: conda create -n landmark 
3. Install the requirements.txt:cd /landmark-attention-qlora/llama; pip install -r requirements.txt


## Training a new model
Adjust the model_name_or_path, output_dir, and cache_dir.
```
python train_qlora.py  
    --model_name_or_path /home/ubuntu/models/wizardLM-7B-HF 
    --output_dir /home/ubuntu/models/wizardLM-7B-HF/lora 
    --cache_dir /home/ubuntu/hf-cache 
    --per_device_train_batch_size 16     
    --gradient_accumulation_steps 8     
    --learning_rate 0.00015     
    --weight_decay 0.1     
    --logging_steps 1     
    --warmup_ratio 0.03 
    --max_steps 200 
    --bf16 False 
    --tf32 False 
    --group_by_length True 
    --lora_r 64 
    --lora_alpha 16 
    
   
```
## Merging
To be able to use the QLoRA, you need to merge it with the base model. This just adds more weights the base model weights. 
```
python merge_peft.py   
    --base_model_name_or_path <base_model_path> 
    --peft_model_path <QLoRA_path> 
    --output_dir <merged_output_path> 
```

## Testing
The merged models can be tested using llama/run_test.py or oobooga. When using with oobooga, the --trust-remote-flag code has to be enabled for the memory to function correctly. The text UI seems to do worse than the run_test.py at remember the correct context, so the code is still being looked into to identify the issue.

python /llama/run_test.py < merged_model_path >
