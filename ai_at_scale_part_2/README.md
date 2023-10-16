# AI for Science at Scale -- Part 2

Machine learning (ML) is a subset of Artificial Intelligence (AI) that uses statistical learning algorithms to build applications that have the ability to automatically learn and improve from its experiences. Most of us use ML in our day to day life when we use services like search engines, voice assistants, and recommendations on Netflix. In ML, an algorithm is trained by providing it with a significant amount of data and allowing it to learn more about the processed information.

Deep learning (DL) is a subset of ML that is inspired by the way a human brain filters information (like recognizing patterns and identifying objects). Since DL processes information in a similar manner as a human brain does, it is mostly used in applications that people generally perform (e.g., driverless cars being able to recognize a stop sign or distinguish a pedestrian from another object).

From a science point of view, both ML and DL can be applied to various scientific domains to analyze large datasets, handle noise correction, deal with error classification, and classify features in data.

As ML/DL models evolve to keep up with the complexity of the real world, a supercomputer’s resources get more and more valuable. In high-performance computing (HPC), ML/DL is getting more and more popular because of the sheer amount of data that needs to be processed and the computational power it requires.

> This training series spans multiple parts and training events -- details and materials for each different training event are separated below.

## Part 2: AI for Science at Scale - Scaling Out

* Recording: https://vimeo.com/873844751
* Slides: [AI for Science at Scale - Part 2](https://www.olcf.ornl.gov/wp-content/uploads/AIforSciencePart2.pdf)

### Workflow for running on Frontier:

```bash
cd $MEMBERWORK/trn018
git clone https://github.com/olcf/ai-training-series
cd ai-training-series/ai_at_scale_part_2
mkdir logs
sbatch launch_script.frontier
```

## Preparing Oscar dataset

bash download_oscar.sh

## Download Models and Tokenizers

bash dl_models.sh

## DDP Example: Running GPT-XL (1.5B) model on Oscar dataset

sbatch launch_gpt_srun.frontier


## Sharded Data Parallelism Example with DeepSpeed ZeRO

sbatch launch_gptJ_srun.frontier 

## Sharded Data Parallelism Example with PyTorch FSDP 

sbatch launch_gptJ_fsdp.frontier 



## Megatron-DeepSPeed 3D parallelism to train a 22B model
```
cd Megatron-DeepSpeed-ORNL
sbatch launch_gpt22b_srun.frontier
```
