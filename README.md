conda create --prefix /nfs/stak/users/frenchp/hpc-share/.conda/envs/dolphin-v2 python=3.12 \
conda activate dolphin-v2 \
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia \
pip install -r requirements.txt \
