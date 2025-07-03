#!/bin/bash
#SBATCH --job-name=CLIP-MoLE_oxford_pets_1shots_2experts
#SBATCH --output=logs_scripts/mole/CLIP-MoLE_oxford_pets_1shots_2experts.out
#SBATCH --error=error_scripts/mole/CLIP-MoLE_oxford_pets_1shots_2experts.err
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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
--dataset oxford_pets \
--seed 1 \
--shots 1 \
--save_path weights \
--num_experts 2 \
--filename "CLIP-MoLE_oxford_pets"
    