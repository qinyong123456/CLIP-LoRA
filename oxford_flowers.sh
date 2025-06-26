#!/bin/bash
#SBATCH --job-name=cliplora_flowers
#SBATCH --time=02:00:00               
#SBATCH --output=logs_scripts/oxford.out
#SBATCH --error=error_scripts/oxford.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --account=def-leszek        

module load python/3.10.13

source ~/projects/def-leszek/pedro36/envs/CLIP-LoRA-venv/bin/activate

cd ~/projects/def-leszek/pedro36/workspace/CLIP-LoRA

python main.py \
  --root_path /home/pedro36/projects/def-leszek/pedro36/datasets/DATA \
  --dataset oxford_flowers \
  --seed 1 \
  --save_path results/weights \
  --filename "CLIP-LoRA_oxford_flowers"
