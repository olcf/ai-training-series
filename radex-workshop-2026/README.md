# Radex Workshop 2026

Users of the workshop can get access to all the repositories which contain the versions of
radex, RHAPSODY, and rhapsody-openfoam-plugins used for the workshop. These also include the
examples which will be used during the workshop. Note to get the contents of these repositories
this one must be cloned recursively:

```bash
git clone --recursive https://github.com/CrayLabs/radex-workshop-2026.git
```

If you cloned down without recursive:
```bash
cd radex-workshop-2026
git submodule update --init
```

The examples shown in the workshop can be found in:
- [repos/radex/example](https://github.com/radical-cybertools/radex/tree/64cba68d1952c79ebb502e9134686161642bd23b/example/)
- [repos/rhapsody-openfoam-plugin/examples](https://github.com/CrayLabs/rhapsody-plugins-openfoam/tree/5ac3bbc7e2e928522223c38f8f75c845a21d3a4e/examples/)

Please follow the directions for running at your site:

- [OLCF Frontier & Odo](#running-on-olcf-frontier--odo)
- [ALCF Aurora](#running-on-alcf-aurora)
- [NERSC Perlmutter](#running-on-nersc-perlmutter)


## Running on OLCF Frontier & Odo

### Environment Setup

In order to set up your RHAPSODY/Dragon environment for Frontier, you need to execute the following:

```bash
source /lustre/orion/stf007/world-shared/atramirez/rhapsody-training/environment.sh
```

For Odo, execute the following:

```bash
source /gpfs/wolf2/olcf/stf007/world-shared/rhapsody/environment.sh
```

### Running the Examples

1. Get a compute node on Frontier or Odo with the following `salloc` command (one node is fine for the demo)

Frontier
```bash
# optional reservation for registered users
salloc --reservation=rhapsody -N1 -t 1:00:00 -A <your_project>

# standard
salloc -N1 -t 1:00:00 -A <your_project> 
```

Odo
```bash
salloc -N1 -t 1:00:00 -A <your_project> 
```

2. Set up the environment

Frontier
```bash
source /lustre/orion/stf007/world-shared/atramirez/rhapsody-training/environment.sh
```

Odo
```bash
source /gpfs/wolf2/olcf/stf007/world-shared/rhapsody/environment.sh
```

3. To run the `pitzDaily-optimize` workflow, clone the RHAPSODY OpenFOAM plugin to your preferred location and launch the workflow with `dragon`. When running on a single node, it is recommended to use the `-s` option for the Dragon launcher.

For both Frontier & Odo
```bash
git clone https://github.com/CrayLabs/rhapsody-plugins-openfoam rhapsody-plugins-openfoam
cd rhapsody-plugins-openfoam/examples/pitzDaily-optimize/
dragon -s -- driver-staged.py
```

You should expect to see something similar to the following output
```
Iteration 1: best parameters epsilon=30.8103, Cmu=0.138893, C1=1.02924, C2=1.88354; converged=True, final_step=346, avg_inlets=-2.36112, loss=0.212634
Iteration 2: best parameters epsilon=25.7455, Cmu=0.124644, C1=1.02196, C2=2.05247; converged=True, final_step=287, avg_inlets=-2.03995, loss=0.0195871
Iteration 3: best parameters epsilon=25.7455, Cmu=0.124644, C1=1.02196, C2=2.05247; converged=True, final_step=287, avg_inlets=-2.03995, loss=0.0195871
Iteration 4: best parameters epsilon=24.751, Cmu=0.141403, C1=1.02382, C2=2.02487; converged=True, final_step=259, avg_inlets=-1.84032, loss=0.00356132
Iteration 5: best parameters epsilon=24.751, Cmu=0.141403, C1=1.02382, C2=2.02487; converged=True, final_step=259, avg_inlets=-1.84032, loss=0.00356132
Iteration 6: best parameters epsilon=59.42, Cmu=0.1328, C1=1.01623, C2=2.00049; converged=True, final_step=352, avg_inlets=-1.91198, loss=0.000143552
```


## Running on ALCF Aurora

### Environment Setup

In order to set up your RHAPSODY/Dragon environment for Aurora, source the following script.

```bash
source /flare/alcf_training/rhapsody/training_material/env_setup.sh
```

The contents of the script are also copied below

```bash
# Access venv with Rhapsody, DragonHPC and SmartSim/SmartRedis
module load frameworks
source /flare/alcf_training/rhapsody/training_material/venv/bin/activate
export PATH=/flare/alcf_training/rhapsody/training_material/SmartSim/smartsim/_core/bin:$PATH
export SR_LOG_FILE=stdout
export SR_LOG_LEVEL=QUIET

# Access OpenFOAM installation and the Rhapsody plugin
source /flare/catalyst/world_shared/bramesh/codes/OpenFOAM/OpenFOAM-v2606/etc/bashrc.aurora
export FOAM_USER_LIBBIN=/flare/alcf_training/rhapsody/training_material/OpenFOAM-v2606
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/flare/alcf_training/rhapsody/training_material/OpenFOAM-v2606
export OPENFOAM_CASE_DIR=/flare/alcf_training/rhapsody/training_material/OpenFOAM-v2606/rhapsody-plugins-openfoam/examples/openfoam-cases
```

### Running the Examples

1. Get a compute node on Aurora with the following qsub command (one node is fine for the demo)

```bash
qsub -I -q R8794691 -A alcf_training -l select=1,walltime=01:00:00,filesystems=home:flare
```

2. Set up the environment

```bash
source /flare/alcf_training/rhapsody/training_material/env_setup.sh
``` 

3. To run the `pitzDaily-optimize` workflow, copy the clean example directory available on `/flare/alcf_training/rhapsody` to your preferred location and launch the workflow with `dragon`. When running on a single node, it is recommended to use the `-s` option for the Dragon launcher.

```bash
cp -r /flare/alcf_training/rhapsody/training_material/pitzDaily-optimize /path/to/preferred/location
cd /path/to/preferred/location/pitzDaily-optimize
dragon -s driver-staged.py
```

You should expect to see something similar to the following output

```
Iteration 1: best parameters epsilon=30.8103, Cmu=0.138893, C1=1.02924, C2=1.88354; converged=True, final_step=346, avg_inlets=-2.36112, loss=0.212634
Iteration 2: best parameters epsilon=25.7455, Cmu=0.124644, C1=1.02196, C2=2.05247; converged=True, final_step=287, avg_inlets=-2.03995, loss=0.0195871
Iteration 3: best parameters epsilon=25.7455, Cmu=0.124644, C1=1.02196, C2=2.05247; converged=True, final_step=287, avg_inlets=-2.03995, loss=0.0195871
Iteration 4: best parameters epsilon=24.751, Cmu=0.141403, C1=1.02382, C2=2.02487; converged=True, final_step=259, avg_inlets=-1.84032, loss=0.00356132
Iteration 5: best parameters epsilon=24.751, Cmu=0.141403, C1=1.02382, C2=2.02487; converged=True, final_step=259, avg_inlets=-1.84032, loss=0.00356132
Iteration 6: best parameters epsilon=59.42, Cmu=0.1328, C1=1.01623, C2=2.00049; converged=True, final_step=352, avg_inlets=-1.91198, loss=0.000143552
```

## Running on NERSC Perlmutter

### Environment Setup

In order to set up your RHAPSODY/Dragon environment for Perlmutter, source the following script.

```bash
source /global/common/software/nstaff/lisa/training/smartsim/environment.sh
```

### Running the Examples

1. Get a compute node on Perlmutter with the following salloc command (one node is fine for the demo)

```bash
salloc -A ntrain2 -C cpu -q shared -N 1 -c 2 -t 00:30:00 --reservation=smartsim_training
```

*Note* If you need an exclusive CPU node for an example, you can request a different queue, for example
```bash
-q regular
```

2. To run the `pitzDaily-optimize` workflow, follow the steps:

```bash
git clone https://github.com/CrayLabs/rhapsody-plugins-openfoam rhapsody-plugins-openfoam
cd rhapsody-plugins-openfoam/examples/pitzDaily-optimize/
dragon -s -- ./driver-staged.py
```

You should expect to see something similar to the following output

```
Iteration 1: best parameters epsilon=30.8103, Cmu=0.138893, C1=1.02924, C2=1.88354; converged=True, final_step=346, avg_inlets=-2.36112, loss=0.212634
Iteration 2: best parameters epsilon=25.7455, Cmu=0.124644, C1=1.02196, C2=2.05247; converged=True, final_step=287, avg_inlets=-2.03995, loss=0.0195871
Iteration 3: best parameters epsilon=25.7455, Cmu=0.124644, C1=1.02196, C2=2.05247; converged=True, final_step=287, avg_inlets=-2.03995, loss=0.0195871
Iteration 4: best parameters epsilon=24.751, Cmu=0.141403, C1=1.02382, C2=2.02487; converged=True, final_step=259, avg_inlets=-1.84032, loss=0.00356132
Iteration 5: best parameters epsilon=24.751, Cmu=0.141403, C1=1.02382, C2=2.02487; converged=True, final_step=259, avg_inlets=-1.84032, loss=0.00356132
Iteration 6: best parameters epsilon=59.42, Cmu=0.1328, C1=1.01623, C2=2.00049; converged=True, final_step=352, avg_inlets=-1.91198, loss=0.000143552
```

*Note* When running on a single node, it is recommanded to use the `-s` option for the Dragon launcher, as seen in the example above.
