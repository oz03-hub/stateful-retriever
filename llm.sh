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

module add cuda/11.8 #12.1
module add conda/latest

conda activate myenv

PORT=9000
FQDN=$(hostname -f)

echo "Hostname: $(hostname)"
echo "FQDN:     ${FQDN}"
echo "Reachable at: http://${FQDN}:${PORT}"

python3 llm.py --port "${PORT}"
# Deactivate the virtual environment

conda deactivate
