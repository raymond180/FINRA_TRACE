#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=72:00:00
#SBATCH --qos=batch
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=24
#SBATCH --mem 16gb

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ~/miniconda3/bin/activate

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v1 Dc_v1 Dc_v1 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v2 Dc_v2 Dc_v2 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v3 Dc_v3 Dc_v3 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Tc_v1 Tc_v1 Tc_v1 250 &

wait

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v1 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v2 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v3 250 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Tc_v1 250 &

wait
