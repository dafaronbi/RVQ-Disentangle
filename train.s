#!/bin/bash
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=512GB
#SBATCH --job-name=TEST_DDP_DISENTANGLE
#SBATCH --mail-type=END
#SBATCH --mail-user=danielfaronbi@nyu.edu
#SBATCH --output=logs/test_disentangle_%j.out

singularity exec --nv \
--overlay $ovl --overlay n_train.sqf:ro --overlay n_valid.sqf:ro --overlay n_test.sqf:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
 /bin/bash -c "nvidia-smi; source /ext3/env.sh; conda activate m_gen; python -u train.py $com" 