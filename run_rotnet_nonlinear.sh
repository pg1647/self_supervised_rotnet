#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
##SBATCH --partition=gpu
#SBATCH --job-name=nonlinfinetuned
#SBATCH --output=results/slurm_files/slurm_rotnet_nonlinear_%j.out


# when the job ends, send me an email at this email address.
#SBATCH --mail-type=END
#SBATCH --mail-user=pg1647@nyu.edu

module purge
module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32
module load anaconda3/5.3.1
source activate pytorchenv



## --suffix if need to run multiple times for same dataset then create differently named result folders
## results/[dataset]_base_classifier[suffix]/



python rotnet_nonlinear_finetuned.py --nins=$1 --batch_size=128 --epochs=100 --layer=$2 --rot_model_type='_best_acc'
