#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=internvl-1b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:1
#SBATCH --time=4:00:00
#SBATCH --mem=21G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/InternVL2-1B
python Predictor.py --root ../figure-to-data --output ../output/InternVL2-1B --model OpenGVLab/InternVL2-1B