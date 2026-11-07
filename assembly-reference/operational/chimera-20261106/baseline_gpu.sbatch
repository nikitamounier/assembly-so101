#!/usr/bin/env bash
#SBATCH --partition=quick_preemptible
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
set -uo pipefail
renew() { trap - USR1; scontrol requeue "$SLURM_JOB_ID"; }
trap renew USR1
sleep infinity &
wait $!
