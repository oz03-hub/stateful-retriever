#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=96G  # Requested Memory
#SBATCH -p superpod-a100  # Partition
#SBATCH -t 3-12:00:00  # Wall Time
#SBATCH -G 2  # Number of GPUs
#SBATCH -o logs/log-%j.out  # %j = job ID
#SBATCH -e logs/log-%j.err  # %j = job ID
#SBATCH --account=pi_hzamani_umass_edu

module add cuda/13.1  # must be >=12 to match the cu13 torch/flashinfer in the env; 11.8 breaks flashinfer JIT
module add conda/latest

conda activate myenv

# Ask the OS for a free ephemeral port so concurrent/leftover jobs don't collide on 9000
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
FQDN=$(hostname -f)

echo "Hostname: $(hostname)"
echo "FQDN:     ${FQDN}"
echo "Reachable at: http://${FQDN}:${PORT}"

python3 llm.py --port "${PORT}"
# Deactivate the virtual environment

conda deactivate
