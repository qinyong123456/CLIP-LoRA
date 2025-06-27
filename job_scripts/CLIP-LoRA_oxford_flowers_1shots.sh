#!/bin/bash
#SBATCH --job-name=CLIP-LoRA_oxford_flowers_1shots
#SBATCH --output=logs_scripts/CLIP-LoRA_oxford_flowers_1shots.out
#SBATCH --error=error_scripts/CLIP-LoRA_oxford_flowers_1shots.err
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision torchaudio ftfy scipy regex tqdm gdown pandas
export TQDM_DISABLE=1

PYTHONWARNINGS="ignore" python3 main.py \
--root_path /home/pedro36/projects/def-leszek/pedro36/datasets/DATA \
--dataset oxford_flowers \
--seed 1 \
--shots 1 \
--save_path weights \
--filename "CLIP-LoRA_oxford_flowers"
    