#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=internvl2-76b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:4
#SBATCH --time=04:00:00
#SBATCH --mem=320G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/InternVL2-76B
python Predictor.py --root ../figure-to-data --output ../output/InternVL2-76B --model OpenGVLab/InternVL2-Llama3-76B