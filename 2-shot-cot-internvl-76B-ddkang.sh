#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=2-internvl-76b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=ddkang
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:4
#SBATCH --time=24:00:00
#SBATCH --mem=320G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

conda activate internvl-76b-env

mkdir ../output/InternVL2-76B-2-Shot-COT-ddkang
python Predictor.py --root ../figure-to-data --output ../output/InternVL2-76B-2-Shot-COT-ddkang --model OpenGVLab/InternVL2-Llama3-76B --Prompt cot --shot 2
