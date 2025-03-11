#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=TRAIN_TEST
#SBATCH --mail-type=END
#SBATCH --mail-user=danielfaronbi@nyu.edu
#SBATCH --output=logs/train_test_output_%j.out
#SBATCH --error=logs/train_test_error_%j.out

singularity exec --nv \
--overlay $ovl  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
 /bin/bash -c "nvidia-smi; source /ext3/env.sh; conda activate m_gen; python train_new.py -p training_parameters/test_2.yml" 