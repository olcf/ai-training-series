# AI for Science at Scale

Machine learning (ML) is a subset of Artificial Intelligence (AI) that uses statistical learning algorithms to build applications that have the ability to automatically learn and improve from its experiences. Most of us use ML in our day to day life when we use services like search engines, voice assistants, and recommendations on Netflix. In ML, an algorithm is trained by providing it with a significant amount of data and allowing it to learn more about the processed information.

Deep learning (DL) is a subset of ML that is inspired by the way a human brain filters information (like recognizing patterns and identifying objects). Since DL processes information in a similar manner as a human brain does, it is mostly used in applications that people generally perform (e.g., driverless cars being able to recognize a stop sign or distinguish a pedestrian from another object).

From a science point of view, both ML and DL can be applied to various scientific domains to analyze large datasets, handle noise correction, deal with error classification, and classify features in data.

As ML/DL models evolve to keep up with the complexity of the real world, a supercomputer’s resources get more and more valuable. In high-performance computing (HPC), ML/DL is getting more and more popular because of the sheer amount of data that needs to be processed and the computational power it requires.

> This training series spans multiple parts and training events -- details and materials for each different training event are separated below.

## Part 1: AI for Science at Scale - Introduction

* Recording: https://vimeo.com/836918490
* Slides: [AI for Science at Scale - Introduction](https://www.olcf.ornl.gov/wp-content/uploads/AI-For-Science-at-Scale-Introduction.pdf)

### Workflow for running on Ascent:

```bash
cd $MEMBERWORK/trn018
git clone https://github.com/olcf/ai-training-series
cd ai-training-series/ai_at_scale
mkdir logs
bsub cnn-cpu-job.lsf
```

Note, although `cnn-cpu-job.lsf` was used in the above code, you can replace that LSF script with any of the following job scripts:

* `cnn-cpu-job.lsf`: Trains a custom CNN using 1 CPU on 1 node
* `cnn-gpu-job.lsf`: Trains a custom CNN using 1 GPU on 1 node
* `cnn-multigpu-job.lsf`: Trains a custom CNN using 6 GPUs on 1 node 
* `multigpu-job.lsf`: Trains a specific CNN (ResNet-50) using 6 GPUs on 1 node
