#!/bin/bash
#SBATCH -J dolphin-math    # name of my job
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH -o /nfs/stak/users/frenchp/logs/jupyter/dolphin-server.out        # name of output file for batch script
#SBATCH -e /nfs/stak/users/frenchp/logs/jupyter/dolphin-server.err        # name of error file for batch script
#SBATCH -c 16             # number of cores per task. Should be #GPU * 4 at least.
#SBATCH --time=1-00:00:00  # time needed for job
#SBATCH --mem=32g        # memory needed for job

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

# run my job
echo "Loading jupyter from:"
~/hpc-share/.conda/envs/dolphin-v2/bin/jupyter notebook --no-browser --port=8080 --notebook-dir=/nfs/stak/users/frenchp/AI506/dolphin_tracker

conda deactivate
date
