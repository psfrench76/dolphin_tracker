conda create --prefix /nfs/hpc/share/dolphin-tracking/.conda/envs/dolphin python=3.12 

** Edit ~/.condarc to include env and package dirs for easy access **

conda activate /nfs/hpc/share/dolphin-tracking/.conda/envs/dolphin \
module load cuda/12.4 \
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia \
pip install -r requirements.txt 
