#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=24:00:00
#SBATCH --qos=batch
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=24
#SBATCH --mem 64gb

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

source ~/miniconda3/bin/activate

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 25 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 50 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 75 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 100 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 125 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 150 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 200 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/main_argument.py Dc_v4 Dc_v4 Dc_v4 250 &

wait

export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 25 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 50 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 75 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 100 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 125 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 150 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 200 &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/topic_model_analysis.py Dc_v4 250 &

wait
