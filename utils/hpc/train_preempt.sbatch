#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH -A eecs
#SBATCH -p preempt
#SBATCH --output=/nfs/hpc/share/frenchp/dolphin-tracker/logs/slurm/train_%j.out
#SBATCH --error=/nfs/hpc/share/frenchp/dolphin-tracker/logs/slurm/train_%j.err
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
echo "run_name = $RUN_NAME"
echo "data_name = $RUN_NAME"
echo "hyp_path = $HYP_PATH"
echo "weights_path = $WEIGHTS_PATH"

# load modules needed for job
module load anaconda
module load cuda/12.4

source /usr/local/apps/anaconda/2023.03/etc/profile.d/conda.sh
source /nfs/stak/users/frenchp/.bashrc

conda activate dolphin-v2
conda env list

python -V
which python

/nfs/stak/users/frenchp/hpc-share/.conda/envs/dolphin-v2/bin/python src/train.py --data_name $DATA_NAME --run_name $RUN_NAME --hyp_path $HYP_PATH --weights_path $WEIGHTS_PATH --checkpoint_reload > "/nfs/hpc/share/frenchp/dolphin-tracker/logs/results/train_$SLURM_JOBID.log"
conda deactivate
date