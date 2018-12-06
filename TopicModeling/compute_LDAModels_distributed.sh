#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=08:00:00
#SBATCH --qos=dpart
#SBATCH --nodes=6
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem 23938mb

source ~/miniconda3/bin/activate

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle
export PYRO_LOGFILE=pyro.log
export PYRO_LOGLEVEL=DEBUG

srun -N 1 --ntasks=1 python -m Pyro4.naming -n 0.0.0.0 &

srun python -m gensim.models.lda_dispatcher &
srun python -m gensim.models.lda_dispatcher &
srun python -m gensim.models.lda_dispatcher &
srun python -m gensim.models.lda_dispatcher &

srun -N 1 --ntasks=1 --mem=16gb bash -c 'python -m gensim.models.lda_dispatcher' &

srun python ~/FINRA_TRACE/TopicModeling/main_distributed.py &