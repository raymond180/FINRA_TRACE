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

export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle

srun -N 1 --ntasks=1 python -m Pyro4.naming -n 0.0.0.0 &

srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &

srun -N 1 --ntasks=1 python -m gensim.models.lsi_dispatcher &

srun python ~/FINRA_TRACE/TopicModeling/main_distributed.py

