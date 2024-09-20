#!/bin/bash
#SBATCH -p h100                     # partition to run job on
#SBATCH -A  genraitor               # account to charge runtime to
#SBATCH -J train-xl-raft            # job name
#SBATCH -N 1                        # number of nodes (discrete machines) requested
#SBATCH -t 40:5:0                   # time the job should run for [h:m:s]
#SBATCH -n 2                        # number of tasks requested (cpus per job)
#SBATCH --gres=gpu:1                # generic resource requesting 1 gpu
#SBATCH -o logs/train-xl-%x.%j.out  # error output filename (stderr)
#SBATCH -e logs/train-xl-%x.%j.err  # error output filename (stderr)

export PROJECT="genraitor"
export BASE_MODEL="meta-llama/Meta-Llama-3-8B"

# This looks identical to your ~/.bashrc
source /etc/profile.d/modules.sh
module purge
module load python/3.11.5 cuda/12.1 gcc/10.3.0

# Activate the correct environment and run your experiment file.
cd ${HOME}/${PROJECT}
# pip install .
source .venv/bin/activate
python3 -m ${PROJECT} train:raft \
  -n data/finetuned-xl \
  -m ${BASE_MODEL} \
  -t data/training/hf/xlarge.hf
