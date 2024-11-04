#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=molmo-72b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=ddkang
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=ccc0419
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# mkdir ../output/Molmo-72B
# python Predictor.py --root ../figure-to-data --output ../output/Molmo-72B --model allenai/Molmo-72B-0924 --paper_list 1