#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH -A eecs
#SBATCH -p dgx2,gpu,dgxh
#SBATCH --output=/nfs/stak/users/frenchp/logs/slurm/train_%j.out
#SBATCH --error=/nfs/stak/users/frenchp/logs/slurm/train_%j.err
#SBATCH -c 4             # number of cores per task
#SBATCH --time=1-00:00:00  # time needed for job
#SBATCH --mem=32g        # memory needed for job
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100|a40|rtx8000|h100

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

/nfs/stak/users/frenchp/hpc-share/.conda/envs/dolphin-v2/bin/python src/train.py --data_name $DATA_NAME --run_name $RUN_NAME --hyp_path $HYP_PATH --weights_path $WEIGHTS_PATH > "/nfs/stak/users/frenchp/logs/slurm/train_$SLURM_JOBID.log"
conda deactivate
date