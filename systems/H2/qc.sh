#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=logs/qcmps_%j.log
#SBATCH --error=logs/qcmps_%j.log#SBATCH -J qcmps
#SBATCH --mem=8gb
#SBATCH -w ne-05
#SBATCH -p q_kchfo
##SBATCH --no-requeue

source ~/venvs/qc/bin/activate

python -u -m qcmps ./fcidumps/H2_FCIDUMP --bond_qubits 3 --block_type ACU --layers 2 --global_opt
