# #!/bin/bash

# # Total number of papers
# total_papers=122
# # Number of partitions
# num_partitions=6
# # Papers per partition (rounded up)
# papers_per_partition=$(( (total_papers + num_partitions - 1) / num_partitions ))

# # Loop to create and submit Slurm scripts for each partition
# for ((i=0; i<num_partitions; i++)); do
#     # Calculate start and end paper IDs for this partition
#     start=$(( i * papers_per_partition + 1 ))
#     end=$(( (i + 1) * papers_per_partition ))
#     if (( end > total_papers )); then
#         end=$total_papers
#     fi

#     # Create the paper list for this partition
#     paper_list=$(seq -s " " $start $end)

#     # Generate a unique Slurm script for this partition
#     script_name="job_$i.sh"
#     cat <<EOL > $script_name
# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --job-name=2-$i-qwen-72b
# #SBATCH --mail-user=chuxuan3@illinois.edu
# #SBATCH --mail-type=BEGIN,END
# #SBATCH --partition=secondary
# #SBATCH --ntasks-per-node=3
# #SBATCH --gres=gpu:H100:3
# #SBATCH --time=04:00:00
# #SBATCH --mem=240G

# echo "Checking GPU availability..."
# nvidia-smi
# echo "CUDA availability in PyTorch:"
# python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# mkdir -p ../output/Qwen-72B-2-Shot-COT
# python Predictor.py --root ../figure-to-data --output ../output/Qwen-72B-2-Shot-COT --model Qwen/Qwen2-VL-72B-Instruct --Prompt cot --shot 2 --paper_list $paper_list
# EOL

#     # Submit the Slurm script
#     sbatch $script_name
# done

#!/bin/bash

# Total number of papers
total_papers=122
# Number of partitions
num_partitions=6
# Papers per partition (rounded up)
papers_per_partition=$(( (total_papers + num_partitions - 1) / num_partitions ))

# Loop to create and submit Slurm scripts for each partition
for ((i=0; i<num_partitions; i++)); do
    # Skip the last group (i=5)
    if (( i == num_partitions - 1 )); then
        continue
    fi

    # Calculate start and end paper IDs for this partition
    start=$(( i * papers_per_partition + 1 ))
    end=$(( (i + 1) * papers_per_partition ))
    if (( end > total_papers )); then
        end=$total_papers
    fi

    # Only include the latter half of the range
    halfway=$(( (start + end) / 2 + 1 ))
    paper_list=$(seq -s " " $halfway $end)

    # Generate a unique Slurm script for this partition
    script_name="job_$i.sh"
    cat <<EOL > $script_name
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=2-$i-qwen-72b
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

mkdir -p ../output/Qwen-72B-2-Shot-COT
python Predictor.py --root ../figure-to-data --output ../output/Qwen-72B-2-Shot-COT --model Qwen/Qwen2-VL-72B-Instruct --Prompt cot --shot 2 --types Table --paper_list $paper_list
EOL

    # Submit the Slurm script
    sbatch $script_name
done
