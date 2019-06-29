#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output %j.out.txt
#SBATCH --error %j.out.txt
#SBATCH --time=24:00:00
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem 128000mb

source ~/miniconda3/bin/activate

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/data_analysis.py &

wait
