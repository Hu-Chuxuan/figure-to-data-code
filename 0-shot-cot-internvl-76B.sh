#!/bin/bash

# Total number of papers
total_papers=122
# Number of partitions
num_partitions=4
# Papers per partition (rounded up)
papers_per_partition=$(( (total_papers + num_partitions - 1) / num_partitions ))

# Loop to create and submit Slurm scripts for each partition
for ((i=0; i<num_partitions; i++)); do
    # Calculate start and end paper IDs for this partition
    start=$(( i * papers_per_partition + 1 ))
    end=$(( (i + 1) * papers_per_partition ))
    if (( end > total_papers )); then
        end=$total_papers
    fi

    # Create the paper list for this partition
    paper_list=$(seq -s " " $start $end)

    # Generate a unique Slurm script for this partition
    script_name="job_$i.sh"
    cat <<EOL > $script_name
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=0-$i-internvl-76b
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

mkdir ../output/InternVL2-76B-0-Shot-COT
python Predictor.py --root ../figure-to-data --output ../output/InternVL2-76B-0-Shot-COT --model OpenGVLab/InternVL2-Llama3-76B --Prompt cot --shot 0 --paper_list $paper_list
EOL

    # Submit the Slurm script
    sbatch $script_name
done
