#!/bin/bash
#SBATCH -p dl_shared              # partition to run job on
#SBATCH -A  genraitor             # account to charge runtime to
#SBATCH -J experiments            # job name
#SBATCH -N 1                      # number of nodes (discrete machines) requested
#SBATCH -t 0:5:0                  # time the job should run for [h:m:s]
#SBATCH -n 1                      # number of tasks requested (cpus per job)
#SBATCH --gres=gpu:1              # generic resource requesting 1 gpu
#SBATCH -o slurm.out              # output filename (stdout)
#SBATCH -e slurm.err              # error output filename (stderr)

# This looks identical to your ~/.bashrc
source /etc/profile.d/modules.sh
module purge
module load python/3.11.5 cuda/12.3 tmux/2.3 gcc/10.3.0

# Activate the correct environment and run your experiment file.
cd ${HOME}/genraitor
source .venv/bin/activate
pip install .
python3 -m genraitor "$@"
