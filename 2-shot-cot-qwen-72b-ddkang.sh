#!/bin/bash

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=2-qwen-72b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=ddkang
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:3
#SBATCH --time=24:00:00
#SBATCH --mem=240G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir -p ../output/Qwen-72B-2-Shot-COT-ddkang
python Predictor.py --root ../figure-to-data --output ../output/Qwen-72B-2-Shot-COT-ddkang --model Qwen/Qwen2-VL-72B-Instruct --Prompt cot --shot 2

