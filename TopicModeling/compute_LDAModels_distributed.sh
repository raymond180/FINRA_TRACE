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
#SBATCH --mem=16gb

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

source nfshomes/cchen07/miniconda3/bin/activate

echo "working directory = "$SLURM_SUBMIT_DIR

srun bash -c 'export PYRO_SERIALIZERS_ACCEPTED=pickle'
srun bash -c 'export PYRO_SERIALIZER=pickle'

srun --nodes=1 --ntasks=1 --time=08:00:00 bash -c 'a=`hostname`;python -m Pyro4.naming -n 127.0.0.0' &

srun bash -c 'python -m gensim.models.lda_worker --host 127.0.0.0' &
srun bash -c 'python -m gensim.models.lda_worker --host 127.0.0.0' &
srun bash -c 'python -m gensim.models.lda_worker --host 127.0.0.0' &
srun bash -c 'python -m gensim.models.lda_worker --host 127.0.0.0' &

srun --nodes=1 --ntasks=1 --time=08:00:00 --mem=16gb bash -c 'python -m gensim.models.lda_dispatcher --host 127.0.0.0' &

srun python nfshomes/cchen07/FINRA_TRACE/TopicModeling/main_distributed.py