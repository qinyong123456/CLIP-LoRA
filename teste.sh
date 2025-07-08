#!/bin/bash
#SBATCH --job-name=MoleTeste
#SBATCH --output=Moleteste.out
#SBATCH --error=Moleteste.err
#SBATCH --mem=32G
#SBATCH --time=00:20:00
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
--num_experts 2 \
--topk 2 \
--lambda_balance 0.1 \
--save_path weights \
--filename "CLIP-MoLE_oxford_pets"
    
