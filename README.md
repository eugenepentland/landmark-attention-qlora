# Landmark Attention QLoRA

This is an experiemental repo attempting to create a landmark attention model using QLoRA.

**The concept has been validated, but further training is still required. The models do not work in oobooga yet. Evaluation must be run in the run_test.py file until the remote code has been written to allow it work in oobooga.**

A 7B model has been trained using 29GB of VRAM, for 200 steps in 2 hours. It was functional all the way up to 7k context, but the accuracy was only about 60%. Further training should improve these results. Evaluation is still in progress and new models are currently being trained.

Testing so far has been run on a linux system with a RTX Quadro 8000 48GB VRAM graphics card.

## Setup
1. Clone the repo: git clone https://github.com/eugenepentland/landmark-attention-qlora.git
2. Create a Conda enviroment: conda create -n landmark 
3. Install the requirements.txt: cd /landmark-attention-qlora/llama; pip install -r requirements.txt
4. Train a model using train_qlora.py
5. Merge with the original weights using merge_peft.py
6. Evaluate using run_test.py


## Training

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
    --bf16 True 
    --tf32 True 
    --group_by_length True 
    --lora_r 64 
    --lora_alpha 16 
    
   
```
## Merging
Merging typically provides faster inference times than using the QLoRA seperatley.
```
python merge_peft.py   
    --base_model_name_or_path <base_model_path> 
    --peft_model_path <QLoRA_path> 
    --output_dir <merged_output_path> 
```

## Testing
Currently the model can only be used with llama/run_test.py or Axolotl. Oooboga support will be coming inthe near future.

python run_test.py < merged_model_path >