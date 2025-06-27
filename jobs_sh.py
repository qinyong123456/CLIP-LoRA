import os
import itertools

methods = [
"CLIP-LoRA"
]
datasets = [
"oxford_pets",
"oxford_flowers"
]
shots_list = [1, 2]
seeds = [1]

job_scripts_dir = "job_scripts"
logs_dir = "logs_scripts"
error_dir = "error_scripts"
results_dir = "results"
os.makedirs(job_scripts_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

for method, dataset, shots, seed in itertools.product(methods, datasets, shots_list, seeds):
    rank = 2
    alpha = 16
    job_name = f"{method}_{dataset}_{shots}shots"
    script_filename = os.path.join(job_scripts_dir, f"{job_name}.sh")

    script_contents = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/{job_name}.out
#SBATCH --error={error_dir}/{job_name}.err
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

PYTHONWARNINGS="ignore" python3 {method}/main.py \\
--root_path /home/pedro36/projects/def-leszek/pedro36/datasets/DATA \\
--dataset {dataset} \\
--seed {seed} \\
--shots {shots} \\
--save_path weights \\
--filename "{method}_{dataset}"
    """

    with open(script_filename, "w") as f:
        f.write(script_contents)
    os.system(f"sbatch {script_filename}")
