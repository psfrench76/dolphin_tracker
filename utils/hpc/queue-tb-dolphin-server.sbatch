#!/bin/bash
#SBATCH -J tb-dolphin-server    # name of my job
#SBATCH -A eecs
#SBATCH -p dgx2,gpu
#SBATCH -o /nfs/stak/users/frenchp/logs/jupyter/tb-dolphin-server.out        # name of output file for batch script
#SBATCH -e /nfs/stak/users/frenchp/logs/jupyter/tb-dolphin-server.err        # name of error file for batch script
#SBATCH -c 1             # number of cores per task
#SBATCH --time=7-00:00:00  # time needed for job
#SBATCH --mem=32g        # memory needed for job

# gather basic information, can be useful for troubleshooting
hostname
echo $SLURM_JOBID
showjob $SLURM_JOBID

# load modules needed for job
module load anaconda

source /usr/local/apps/anaconda/2023.03/etc/profile.d/conda.sh
source /nfs/stak/users/frenchp/.bashrc

conda activate dolphin-v2

# run my job
echo "Loading tensorboard from:"
which tensorboard
tensorboard --port=6999 --logdir=/nfs/stak/users/frenchp/AI506/dolphin_tracker/runs/ &

# run my job
echo "Loading jupyter from:"
which jupyter
jupyter notebook --no-browser --port=8080 --notebook-dir=/nfs/stak/users/frenchp/AI506/dolphin_tracker

conda deactivate
date
