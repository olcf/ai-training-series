# AI for Science at Scale -- Part 2

Machine learning (ML) is a subset of Artificial Intelligence (AI) that uses statistical learning algorithms to build applications that have the ability to automatically learn and improve from its experiences. Most of us use ML in our day to day life when we use services like search engines, voice assistants, and recommendations on Netflix. In ML, an algorithm is trained by providing it with a significant amount of data and allowing it to learn more about the processed information.

Deep learning (DL) is a subset of ML that is inspired by the way a human brain filters information (like recognizing patterns and identifying objects). Since DL processes information in a similar manner as a human brain does, it is mostly used in applications that people generally perform (e.g., driverless cars being able to recognize a stop sign or distinguish a pedestrian from another object).

From a science point of view, both ML and DL can be applied to various scientific domains to analyze large datasets, handle noise correction, deal with error classification, and classify features in data.

As ML/DL models evolve to keep up with the complexity of the real world, a supercomputer’s resources get more and more valuable. In high-performance computing (HPC), ML/DL is getting more and more popular because of the sheer amount of data that needs to be processed and the computational power it requires.

> This training series spans multiple parts and training events -- details and materials for each different training event are separated below.

## Part 2: AI for Science at Scale - Scaling Out

* Recording: Will be posted after the tutorial
* Slides: Will be posted after the training

### Workflow for running on Frontier:

```bash
cd $MEMBERWORK/trn018
git clone https://github.com/olcf/ai-training-series
cd ai-training-series/ai_at_scale_part_2
mkdir logs
sbatch launch_script.frontier
```

For DDP Examples:
* cd to ddp_examples
* `launch_gpt_srun.frontier`: Trains a custom GPT Model on multiple nodes

For Megatron Examples:
* cd to megatron_examples/Megatron-DeepSpeed-ORNL
* `launch_gpt_srun.frontier`: Trains a custom GPT Model of size 1.4B on multiple nodes
* `launch_gpt175b_srun.frontier`: Trains a custom GPT Model of size 175B on multiple nodes
