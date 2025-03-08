conda create --prefix /nfs/hpc/share/dolphin-tracking/.conda/envs/dolphin python=3.12 

** Edit ~/.condarc to include env and package dirs for easy access if it exists. IF IT DOES NOT EXIST, do: **
cp .condarc ~

conda activate /nfs/hpc/share/dolphin-tracking/.conda/envs/dolphin \
module load cuda/12.4 \
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia \
pip install -r requirements.txt 

mkdir ../saves
mkdir ../saves/pretrained

Copy weights file into ../saves/pretrained

mkdir ../data
mkdir ../data/original_source
mkdir ../data/original_source/raw_videos

Copy an example video into ../data/original_source/raw_videos
