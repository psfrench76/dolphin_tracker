#!/bin/bash
#SBATCH -J long-jtb-server    # name of my job
#SBATCH -A eecs
#SBATCH -p dgx2,gpu
#SBATCH -o /nfs/stak/users/frenchp/logs/jupyter/long-jtb-server.out        # name of output file for batch script
#SBATCH -e /nfs/stak/users/frenchp/logs/jupyter/long-jtb-server.err        # name of error file for batch script
#SBATCH -c 1             # number of cores per task
#SBATCH --time=7-00:00:00  # time needed for job
#SBATCH --mem=10g        # memory needed for job
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100|a40|rtx8000|h100

# gather basic information, can be useful for troubleshooting
hostname
echo $SLURM_JOBID
showjob $SLURM_JOBID

# load modules needed for job
module load anaconda
module load cuda

source /usr/local/apps/anaconda/2023.03/etc/profile.d/conda.sh
source /nfs/stak/users/frenchp/.bashrc

source /nfs/stak/users/frenchp/osu-configs/conda-select.sh
activate_conda_env

# run my job
echo "Loading tensorboard from:"
which tensorboard
tensorboard --port=6999 --logdir=/nfs/stak/users/frenchp/tbruns/ &

# run my job
echo "Loading jupyter from:"
which jupyter
jupyter notebook --no-browser --port=8080 --notebook-dir=/nfs/stak/users/frenchp

conda deactivate
date
