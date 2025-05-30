#!/usr/bin/env bash


# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH -A cil
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --time=59:59
#SBATCH --mem-per-cpu=2048
#SBATCH --output=run_seed_{$1}_nce_{$2}.out
#SBATCH --error=run_seed_{$1}_nce_{$2}.err

module load cuda/12.6.0
source /cluster/courses/cil/envs/miniforge3/etc/profile.d/conda.sh

conda activate /cluster/courses/cil/envs/collaborative_filtering/
pip install tqdm
python /home/shgoel/CIL/cil-collaborative-filtering/two_tower.py --seed $1 --nce $2
EOT

sbatch < job.sh
rm job.sh