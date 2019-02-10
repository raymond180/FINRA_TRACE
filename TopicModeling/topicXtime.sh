#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=24:00:00
#SBATCH --qos=batch
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=24
#SBATCH --mem 32gb


source ~/miniconda3/bin/activate

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Dc_v1 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Dc_v2 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Dc_v3 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Tc_v1 250 &

wait

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Dc_v1 50 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Dc_v2 50 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Dc_v3 50 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topicXtime_main.py Tc_v1 50 &

wait