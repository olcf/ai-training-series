# AI for Science at Scale -- Part III
In Part 3 of this training series, we will demonstrate how to train LLMs with hundreds of billions of parameters from scratch. We will utilize two of our external repostitories/branches for this purpose.

## Training Very Large LLMs
### Get Megatron-DeepSpeed Codebase Ported to Frontier
```
git clone https://github.com/sajal-vt/Megatron-DeepSpeed-ORNL.git
git fetch
git switch FA2
```

### Training a Model with 22 Billion Parameters on 2 Nodes
```
sbatch --reservation=ai launch_gpt22b_bf16.slurm
```

### Training a Model with 175 Billion Parameters on 16 Nodes
```
sbatch --reservation=ai launch_gpt175b_bf16.slurm
```

### Training a Model with 1 Trillion Parameters on 128 Nodes
```
sbatch --reservation=ai launch_gpt1T_bf16.slurm
```

## Finding the Best Distributed Training Strategy using DeepHyper
### Get frontier-sd Branch
```
git clone https://github.com/sajal-vt/Megatron-DeepSpeed-ORNL.git
git fetch
git switch frontier-sd
```

### Launch HyperParameter Search using DeepHyper
```
sbatch --reservation=ai launch_dh.frontier
```


