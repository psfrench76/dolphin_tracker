conda create --prefix /nfs/stak/users/frenchp/hpc-share/.conda/envs/dolphin-v1 python=3.12
conda activate dolphin-v1
conda install pytorch torchvision pytorch-cuda=12.4 ultralytics -c pytorch -c nvidia -c conda-forge
pip install -r requirements.txt
module load cuda/12.4
