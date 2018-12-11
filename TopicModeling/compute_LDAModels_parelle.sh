#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=08:00:00
#SBATCH --qos=dpart
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --mem 16gb

source ~/miniconda3/bin/activate
srun python ~/FINRA_TRACE/TopicModeling/main.py