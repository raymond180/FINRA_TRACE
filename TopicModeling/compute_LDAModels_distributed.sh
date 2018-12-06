#!/bin/bash

#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=08:00:00
#SBATCH --qos=batch
#SBATCH --nodes=6
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb

source /nfshomes/cchen07/miniconda3/bin/activate

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

srun bash -c 'export PYRO_SERIALIZERS_ACCEPTED=pickle'
srun bash -c 'export PYRO_SERIALIZER=pickle'

#bash -c 'a=`hostname`;python -m Pyro4.naming -n $a' &

srun --nodes=1 --ntasks=1 --time=08:00:00 bash -c 'a=`hostname`;python -m Pyro4.naming -n 0.0.0.0' &

srun bash -c 'python -m gensim.models.lda_worker --host $a' &
srun bash -c 'python -m gensim.models.lda_worker --host $a' &
srun bash -c 'python -m gensim.models.lda_worker --host $a' &
srun bash -c 'python -m gensim.models.lda_worker --host $a' &

srun --nodes=1 --ntasks=1 --time=08:00:00 --mem=16gb bash -c 'python -m gensim.models.lda_dispatcher --host $a' &

srun python /nfshomes/cchen07/FINRA_TRACE/TopicModeling/main_distributed.py