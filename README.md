conda create --prefix /nfs/hpc/share/dolphin-tracking/.conda/envs/dolphin python=3.12 

** Edit ~/.condarc to include env and package dirs for easy access if it exists. IF IT DOES NOT EXIST, do: ** \
cp .condarc ~

conda activate /nfs/hpc/share/dolphin-tracking/.conda/envs/dolphin \
module load cuda/12.4 \
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia \
pip install -r requirements.txt 

Set up automatic group permissions for storage root: \
setfacl -d -m "u::rwx,g::rwx,o::-" .. \
chmod g+w -R ../data \
chmod g+w -R ../output 

Set up automatic permissions for project root: \
setfacl -d -m "u::rwx,g::rx,o::-" .

Set up automatic permissions for cfg and usr dirs: \
setfacl -d -m "u::rwx,g::rwx,o::-" cfg \
setfacl -d -m "u::rwx,g::rwx,o::-" usr \
chmod g+w -R usr/ \
chmod g+w -R cfg/

mkdir ../saves \
mkdir ../saves/pretrained

Copy weights file into ../saves/pretrained

mkdir ../data \
mkdir ../data/original_source \
mkdir ../data/original_source/raw_videos

Copy an example video into ../data/original_source/raw_videos
