#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=08:00:00
#SBATCH --qos=dpart
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem 7821mb

export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle

source ~/miniconda3/bin/activate

srun -N 1 python -m Pyro4.naming -n 0.0.0.0 &

srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &
srun python -m gensim.models.lsi_worker &

srun -N 1 --mem=20gb python -m gensim.models.lsi_dispatcher &

srun python ~/FINRA_TRACE/TopicModeling/main_distributed.py

