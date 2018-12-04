#SBATCH --job-name=topicModeling
#SBATCH --output main.out.%j
#SBATCH --error main.out.%j
#SBATCH --time=05:00:00
#SBATCH --qos=dpart
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --mem 16gb

source ~/miniconda3/bin/activate
srun -N 1 --mem=16gb python ~/FINRA_TRACE/TopicModeling/main.py