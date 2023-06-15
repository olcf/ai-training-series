Workflow for running on Ascent:

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
