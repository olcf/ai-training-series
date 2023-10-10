# transformer-on-crusher



## Preparing Oscar dataset

bash download_oscar.sh

## Download Models and Tokenizers

bash dl_models.sh

## DDP Example: Running GPT-XL (1.5B) model on Oscar dataset

sbatch launch_gpt_srun.frontier


## Sharded Data Parallelism Example with DeepSpeed ZeRO

sbatch launch_gptJ_srun.frontier 

