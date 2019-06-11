#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output %j.out.txt
#SBATCH --error %j.out.txt
#SBATCH --time=24:00:00
#SBATCH --qos=batch
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=24
#SBATCH --mem 100gb

source ~/miniconda3/bin/activate

srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW small 50  &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW large 50  &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW small 100  &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW large 100  &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW small 200  &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW large 200  &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW small 300  &
srun --nodes=1 --ntasks=1 --exclusive python ~/FINRA_TRACE/TopicModeling/compute_lda.py trade_vol_BoW large 300  &

wait