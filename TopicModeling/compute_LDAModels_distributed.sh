#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=08:00:00
#SBATCH --qos=dpart
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem 23938mb

source ~/miniconda3/bin/activate

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

srun bash -c 'export PYRO_SERIALIZERS_ACCEPTED=pickle'
srun bash -c 'export PYRO_SERIALIZER=pickle'
srun bash -c 'export PYRO_LOGFILE=pyro.log'
srun bash -c 'export PYRO_LOGLEVEL=DEBUG'

srun --mem=1gb python -m Pyro4.naming -n 0.0.0.0 &

srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &

srun --mem=32gb python -m gensim.models.lsi_dispatcher &

srun python ~/FINRA_TRACE/TopicModeling/main_distributed.py