# AI for Science at Scale -- Part III
In Part 3 of this training series, we will demonstrate how to train LLMs with hundreds of billions of parameters from scratch. We will utilize two of our external repostitories/branches for this purpose.

## Training Very Large LLMs
### Get Megatron-DeepSpeed Codebase Ported to Frontier
```
git clone https://github.com/sajal-vt/Megatron-DeepSpeed-ORNL.git
cd Megatron-DeepSpeed-ORNL/
git fetch
git switch FA2
```

### Using the batch scripts mentioned below

You must modify the batch scripts due to a permissions issue. You have to replace these 2 lines (top of batch script):
```
source /lustre/orion/world-shared/stf218/sajal/miniconda3-frontier/bin/activate
conda activate ...
```
with:
```
module load miniforge3
source activate ...
```


### Training a Model with 22 Billion Parameters on 2 Nodes
```
sbatch -A your_project_ID --reservation=ai launch_gpt22b_bf16.slurm
```

### Training a Model with 175 Billion Parameters on 16 Nodes
```
sbatch -A your_project_ID --reservation=ai launch_gpt175b_bf16.slurm
```

### Training a Model with 1 Trillion Parameters on 128 Nodes
```
sbatch -A your_project_ID --reservation=ai launch_gpt1T_bf16.slurm
```

## Finding the Best Distributed Training Strategy using DeepHyper
### Get frontier-sd Branch
```
git clone https://github.com/sajal-vt/Megatron-DeepSpeed-ORNL.git
cd Megatron-DeepSpeed-ORNL/
git fetch
git switch frontier-sd
```

### Launch HyperParameter Search using DeepHyper
```
sbatch -A your_project_ID --reservation=ai launch_dh.frontier
```


