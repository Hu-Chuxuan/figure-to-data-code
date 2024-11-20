#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=2-$i-internvl-76b
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

conda init
conda activate internvl-76b-env

mkdir ../output/InternVL2-76B-2-Shot-COT
python Predictor.py --root ../figure-to-data --output ../output/InternVL2-76B-2-Shot-COT --model OpenGVLab/InternVL2-Llama3-76B --Prompt cot --shot 2 --types Table --paper_list 20 21 22 39 40 41 42