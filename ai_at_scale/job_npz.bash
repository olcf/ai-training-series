#BSUB -P stf218
#BSUB -W 2:00
#BSUB -q debug
#BSUB -nnodes 5
#BSUB -J spacegroup-pretrained-npz
#BSUB -o sg-10x5-pretrained-npz%J.o
#BSUB -e sg-10x5-pretrained-npz%J.e

module load open-ce
#conda activate /ccs/home/sajaldash/.conda/envs/newenv
export TORCH_HOME=/gpfs/alpine/world-shared/gen011/sajal/dc20
#export DATA_DIR=/gpfs/alpine/world-shared/gen011/sajal/dc20/spacegroup/mxdata
#export DATA_DIR=/gpfs/alpine/proj-shared/stf011/sajal/gitspace/spacegroup-classifier/data
export DATA_DIR=/gpfs/alpine/world-shared/stf218/sajal/stemdl-data/
jsrun -n 5 --tasks_per_rs 6 --cpu_per_rs 42 --gpu_per_rs 6 --rs_per_host 1 --latency_priority CPU-CPU --launch_distribution packed --bind packed:1 python pir_classifier_npz.py --epochs 10 --batch-size 5 --train-dir $DATA_DIR/train --val-dir $DATA_DIR/test
