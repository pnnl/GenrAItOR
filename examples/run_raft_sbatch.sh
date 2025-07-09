#!/bin/bash

#SBATCH -A genraitor
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -p a100_shared
#SBATCH -J make_raft_dataset
#SBATCH --gres=gpu:1
#SBATCH -o logs/run_raft_sbatch.log
#SBATCH -e logs/run_raft_sbatch_stderr.log

#First make sure the module commands are available.
source /etc/profile.d/modules.sh

#Set up your environment you wish to run in with module commands.
module purge
module load python/miniconda3.9

##### activate conda with roamingbanshee installed #####
source /people/clab683/miniconda3/bin/activate genai
conda list

context_path=${1:-/people/clab683/git_repos/llama-3-raft/data/context_2438_uniprots_2024-09-07:10:39.txt}
python_target=${2:-/people/clab683/miniconda3/envs/genai/bin/python}
repo_dir=${3:-/people/clab683/git_repos/llama-3-raft}

#Next unlimit system resources, and set any other environment variables you need.
ulimit

cd ${repo_dir}

echo "beginning raft-dataset construction"

export PYTHONPATH=./

${python_target} examples/raft-dataset.py \
--context_path $context_path \
--embed local \
> logs/run-raft.log