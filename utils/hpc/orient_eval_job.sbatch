#!/bin/bash
#SBATCH --job-name=orient_eval_job
#SBATCH -A eecs
#SBATCH -p dgx2,gpu,dgxh,preempt
#SBATCH --output=/nfs/hpc/share/frenchp/dolphin-tracker/logs/slurm/orient_eval_%j.out
#SBATCH --error=/nfs/hpc/share/frenchp/dolphin-tracker/logs/slurm/orient_eval_%j.err
#SBATCH -c 4             # number of cores per task
#SBATCH --time=1-00:00:00  # time needed for job
#SBATCH --mem=32g        # memory needed for job
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100|a40|rtx8000|h100
#SBATCH --requeue

# gather basic information, can be useful for troubleshooting
hostname
echo $SLURM_JOBID
showjob $SLURM_JOBID

# load modules needed for job
module load anaconda
module load cuda/12.4

source /usr/local/apps/anaconda/2023.03/etc/profile.d/conda.sh
source /nfs/stak/users/frenchp/.bashrc

conda activate dolphin-v2
conda env list

python -V
which python

/nfs/stak/users/frenchp/hpc-share/.conda/envs/dolphin-v2/bin/python src/eval_orientations.py --dataset $DATASET --output_folder $OUTPUT_FOLDER --weights $WEIGHTS $AUGMENT $FILTER_ANGLES > "/nfs/hpc/share/frenchp/dolphin-tracker/logs/results/orient_eval_$SLURM_JOBID.log"
conda deactivate
date