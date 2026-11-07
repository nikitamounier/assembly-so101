#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
trap 'scontrol requeue "$SLURM_JOB_ID"' USR1
sleep infinity &
wait $!
