#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=llava-7b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:1
#SBATCH --time=00:50:00
#SBATCH --mem=21G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir ../output/Llava-7B
python Predictor.py --root ../figure-to-data --output ../output/Llava-7B --model lmms-lab/llava-onevision-qwen2-7b-ov --paper_list 1 2 3