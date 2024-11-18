#!/bin/bash

# Total number of papers
total_papers=122
# Number of partitions
num_partitions=10
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
#SBATCH --job-name=0-$i-llava-72b
#SBATCH --mail-user=chuxuan3@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=secondary
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:H100:3
#SBATCH --time=04:00:00
#SBATCH --mem=240G

echo "Checking GPU availability..."
nvidia-smi
echo "CUDA availability in PyTorch:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

mkdir -p ../output/Llava-72B-0-Shot-COT-batch
python Predictor.py --root ../figure-to-data --output ../output/Llava-72B-0-Shot-COT-batch --model lmms-lab/llava-onevision-qwen2-72b-si --Prompt cot --shot 0 --paper_list $paper_list
EOL

    # Submit the Slurm script
    sbatch $script_name
done
